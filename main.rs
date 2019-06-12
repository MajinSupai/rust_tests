/*
Dependencies: (Because I didn't include the cargo.toml)
vulkano = "0.11"
vulkano-shaders = "0.11"
image = "0.21"

*/
extern crate vulkano;
extern crate vulkano_shaders;
extern crate image;

/* Vulkano imports */
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;

use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;

use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;

use vulkano::sync::GpuFuture;
use std::sync::Arc;

use vulkano::pipeline::ComputePipeline;

use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use vulkano::format::Format;
use vulkano::format::ClearValue;

use vulkano::image::Dimensions;
use vulkano::image::StorageImage;

/* image imports */
use image::{ImageBuffer, Rgba};

/* Shader to compute the mandelbrot set */
mod mandel_shader {
	vulkano_shaders::shader!{
		ty: "compute",
		src: "
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

void main() {
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    vec2 z = vec2(0.0, 0.0);
    float i;
    for (i = 0.0; i < 1.0; i += 0.005) {
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );

        if (length(z) > 4.0) {
            break;
        }
    }

    vec4 to_write = vec4(vec3(i), 1.0);
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}"
	}
}

/* Convenience struct to hold the Vulkano state */
struct VulkanoState<'a> {
	vulkano : &'a std::sync::Arc<Instance>,
	physical : &'a PhysicalDevice<'a>,
	device : &'a std::sync::Arc<Device>,
	queue : &'a std::sync::Arc<Queue>,
}

/* Test mandelbrot function */
fn test_mandel(state: &VulkanoState) {
	/* Create image buffer on GPU */
	let image = StorageImage::new(
			state.device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
			Format::R8G8B8A8Unorm, Some(state.queue.family()))
		.unwrap();
	
	/* Create output buffer on GPU */
	let buf = CpuAccessibleBuffer::from_iter(
			state.device.clone(), BufferUsage::all(),
			(0 .. 1024 * 1024 * 4).map(|_| 0u8))
		.expect("Failed to create buffer");
	
	let shader = mandel_shader::Shader::load(state.device.clone())
		.expect("Failed to load shader");
	
	let compute_pipeline = Arc::new(ComputePipeline::new(state.device.clone(), &shader.main_entry_point(), &())
		.expect("Failed to create compute pipeline"));
	
	/* Add image to descriptor set */
	let set = Arc::new(PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
		.add_image(image.clone()).unwrap()
		.build().unwrap());
	
	/* Create command buffer which calls shader, then copies result to out buffer*/
	let command_buffer = AutoCommandBufferBuilder::new(state.device.clone(), state.queue.family()).unwrap()
		.dispatch([1024 / 8, 1024 / 8, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
		.copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
		.build().unwrap();
	
	/* Wait for results */
	let finished = command_buffer.execute(state.queue.clone()).unwrap();
	finished.then_signal_fence_and_flush().unwrap()
		.wait(None).unwrap();
	
	/* Retrieve buffer, then save as png */
	let buffer_content = buf.read().unwrap();
	let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
	
	image.save("mandel.png").unwrap();
	
	println!("Mandelbrot created successfully!");
	
}

fn main() {
	/* Setting up vulkano */
	let vulkano = Instance::new(None, &InstanceExtensions::none(), None)
		.expect("Failed to initialize Vulkan");
	
	let physical = PhysicalDevice::enumerate(&vulkano).next()
		.expect("No Vulkan device available");
	
	let queue_family = physical.queue_families()
		.find(|&q| q.supports_graphics())
		.expect("No Vulkan graphics device found");
	
	let (device, mut queues) = {
		Device::new(
			physical, &Features::none(), &DeviceExtensions::none(),
			[(queue_family, 0.5)].iter().cloned()).expect("Failed to create Vulkan device")
	};
	
	let queue = queues.next().unwrap();
	
	/* Create a convenience state object */
	let state = VulkanoState {
		vulkano: &vulkano,
		physical: &physical,
		device: &device,
		queue: &queue
	};
	
	/* Test GPU by creating a mandelbrot set */
	test_mandel(&state);
}