use ocl::{Platform, Device, Context, Queue, DeviceType};

pub struct GpuContext {
    pub context: Context,
    pub queue:   Queue,
    pub device:  Device,
}

impl GpuContext {
    pub fn try_init() -> Option<Self> {
        let platform = Platform::list().into_iter()
            .find(|p| {
                let name = p.name().unwrap_or_default().to_lowercase();
                name.contains("nvidia") || name.contains("amd") || name.contains("intel")
            })
            .or_else(|| Platform::list().into_iter().next())?;

        let device = Device::list(platform, Some(DeviceType::GPU))
            .ok()?
            .into_iter()
            .next()?;

        println!("[GPU] Platform: {}", platform.name().unwrap_or_default());
        println!("[GPU] Device:   {}\n", device.name().unwrap_or_default());

        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .ok()?;

        let queue = Queue::new(&context, device, None).ok()?;

        Some(GpuContext { context, queue, device })
    }
}
