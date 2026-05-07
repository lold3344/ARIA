use wgpu;
use pollster;

const MAX_A_SIZE: usize = 512 * 2048;
const MAX_B_SIZE: usize = 2048 * 16000;
const MAX_OUT_SIZE: usize = 512 * 16000;
const MAX_B_ELEMS: usize = MAX_B_SIZE / 4;

pub struct LSTMGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    matmul_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    buf_c: wgpu::Buffer,
    buf_dims: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl LSTMGpu {
    pub fn try_init() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;

        let info = adapter.get_info();
        println!("Found GPU: {} ({:?})", info.name, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ARIA GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )).ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lstm.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "matmul",
            compilation_options: Default::default(),
        });

        let a_size = MAX_A_SIZE * 4;
        let b_size = MAX_B_SIZE * 4;
        let out_size = MAX_OUT_SIZE * 4;

        let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf_a"),
            size: a_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf_b"),
            size: b_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_c = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf_c"),
            size: out_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_dims = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buf_dims"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        println!("GPU pool: a={}MB b={}MB c={}MB\n",
            a_size / 1024 / 1024, b_size / 1024 / 1024, out_size / 1024 / 1024);

        Some(LSTMGpu {
            device,
            queue,
            matmul_pipeline,
            bind_group_layout,
            buf_a,
            buf_b,
            buf_c,
            buf_dims,
            staging,
        })
    }

    pub fn matmul(&self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32) -> Vec<f32> {
        let out_size = (m * n) as usize;

        // Fall back to CPU if sizes exceed pre-allocated buffer limits
        if a.len() > MAX_A_SIZE || b.len() > MAX_B_ELEMS || out_size > MAX_OUT_SIZE {
            let a2 = ndarray::Array2::<f32>::from_shape_vec(
                (m as usize, k as usize), a.to_vec()
            ).unwrap();
            let b2 = ndarray::Array2::<f32>::from_shape_vec(
                (k as usize, n as usize), b.to_vec()
            ).unwrap();
            let c = a2.dot(&b2);
            return c.into_raw_vec();
        }

        self.queue.write_buffer(&self.buf_a, 0, bytemuck::cast_slice(a));
        self.queue.write_buffer(&self.buf_b, 0, bytemuck::cast_slice(b));
        self.queue.write_buffer(&self.buf_dims, 0, bytemuck::cast_slice(&[m, n, k, 0u32]));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_c.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_dims.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((m + 15) / 16, (n + 15) / 16, 1);
        }
        encoder.copy_buffer_to_buffer(&self.buf_c, 0, &self.staging, 0, (out_size * 4) as u64);

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        let slice = self.staging.slice(..);
        let (sender, receiver) = crossbeam_channel::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |result| { sender.send(result).ok(); });
        receiver.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging.unmap();

        result
    }
}
