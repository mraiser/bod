use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, IndexOp, Result, Tensor, D, Device};
use candle_nn::{batch_norm, conv2d, conv2d_no_bias, Conv2d, Conv2dConfig, Module, VarBuilder};
use candle_transformers::object_detection::{non_maximum_suppression, Bbox, KeyPoint};
use clap::{Parser, ValueEnum};
use image::DynamicImage;

// Keypoints as reported by ChatGPT :)
// Nose
// Left Eye
// Right Eye
// Left Ear
// Right Ear
// Left Shoulder
// Right Shoulder
// Left Elbow
// Right Elbow
// Left Wrist
// Right Wrist
// Left Hip
// Right Hip
// Left Knee
// Right Knee
// Left Ankle
// Right Ankle
const KP_CONNECTIONS: [(usize, usize); 16] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
];
// Model architecture from https://github.com/ultralytics/ultralytics/issues/189
// https://github.com/tinygrad/tinygrad/blob/master/examples/yolov8.py

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Model weights, in safetensors format.
    #[arg(long)]
    model: Option<String>,

    /// Which model variant to use.
    #[arg(long, value_enum, default_value_t = Which::S)]
    which: Which,

    images: Vec<String>,

    /// Threshold for the model confidence level.
    #[arg(long, default_value_t = 0.25)]
    confidence_threshold: f32,

    /// Threshold for non-maximum suppression.
    #[arg(long, default_value_t = 0.45)]
    nms_threshold: f32,

    /// The task to be run.
    #[arg(long, default_value = "detect")]
    task: YoloTask,

    /// The size for the legend, 0 means no legend.
    #[arg(long, default_value_t = 14)]
    legend_size: u32,
}

impl Args {
    fn model(&self) -> anyhow::Result<std::path::PathBuf> {
        let path = match &self.model {
            Some(model) => std::path::PathBuf::from(model),
            None => {
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model("lmz/candle-yolo-v8".to_string());
                let size = match self.which {
                    Which::N => "n",
                    Which::S => "s",
                    Which::M => "m",
                    Which::L => "l",
                    Which::X => "x",
                };
                let task = match self.task {
                    YoloTask::Pose => "-pose",
                    YoloTask::Detect => "",
                };
                api.get(&format!("yolov8{size}{task}.safetensors"))?
            }
        };
        Ok(path)
    }
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub const NAMES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

pub fn report_detect(
    pred: &Tensor,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
    legend_size: u32,
) -> Result<DynamicImage> {
    let pred = pred.to_device(&Device::Cpu)?;
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > confidence_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }
println!("-----------------------------------------------------------------------------------------");
println!("{:?}", bboxes);
println!("-----------------------------------------------------------------------------------------");

    non_maximum_suppression(&mut bboxes, nms_threshold);

    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    let font = Vec::from(include_bytes!("roboto-mono-stripped.ttf") as &[u8]);
    let font = rusttype::Font::try_from_vec(font);
    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
        for b in bboxes_for_class.iter() {
            println!(
                "{}: {:?}",
                NAMES[class_index],
                b
            );
            let xmin = (b.xmin * w_ratio) as i32;
            let ymin = (b.ymin * h_ratio) as i32;
            let dx = (b.xmax - b.xmin) * w_ratio;
            let dy = (b.ymax - b.ymin) * h_ratio;
            if dx >= 0. && dy >= 0. {
                imageproc::drawing::draw_hollow_rect_mut(
                    &mut img,
                    imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                    image::Rgb([255, 0, 0]),
                );
            }
            if legend_size > 0 {
                if let Some(font) = font.as_ref() {
                    imageproc::drawing::draw_filled_rect_mut(
                        &mut img,
                        imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, legend_size),
                        image::Rgb([170, 0, 0]),
                    );
                    let legend = format!(
                        "{}   {:.0}%",
                        NAMES[class_index],
                        100. * b.confidence
                    );
                    imageproc::drawing::draw_text_mut(
                        &mut img,
                        image::Rgb([255, 255, 255]),
                        xmin,
                        ymin,
                        rusttype::Scale::uniform(legend_size as f32 - 1.),
                        font,
                        &legend,
                    )
                }
            }
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

pub fn report_pose(
    pred: &Tensor,
    img: DynamicImage,
    w: usize,
    h: usize,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<DynamicImage> {
    let pred = pred.to_device(&Device::Cpu)?;
    let (pred_size, npreds) = pred.dims2()?;
    if pred_size != 17 * 3 + 4 + 1 {
        candle_core::bail!("unexpected pred-size {pred_size}");
    }
    let mut bboxes = vec![];
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = pred[4];
        if confidence > confidence_threshold {
            let keypoints = (0..17)
                .map(|i| KeyPoint {
                    x: pred[3 * i + 5],
                    y: pred[3 * i + 6],
                    mask: pred[3 * i + 7],
                })
                .collect::<Vec<_>>();
            let bbox = Bbox {
                xmin: pred[0] - pred[2] / 2.,
                ymin: pred[1] - pred[3] / 2.,
                xmax: pred[0] + pred[2] / 2.,
                ymax: pred[1] + pred[3] / 2.,
                confidence,
                data: keypoints,
            };
            bboxes.push(bbox)
        }
    }

    let mut bboxes = vec![bboxes];
    non_maximum_suppression(&mut bboxes, nms_threshold);
    let bboxes = &bboxes[0];

    // Annotate the original image and print boxes information.
    let (initial_h, initial_w) = (img.height(), img.width());
    let w_ratio = initial_w as f32 / w as f32;
    let h_ratio = initial_h as f32 / h as f32;
    let mut img = img.to_rgb8();
    for b in bboxes.iter() {
        println!("{b:?}");
        let xmin = (b.xmin * w_ratio) as i32;
        let ymin = (b.ymin * h_ratio) as i32;
        let dx = (b.xmax - b.xmin) * w_ratio;
        let dy = (b.ymax - b.ymin) * h_ratio;
        if dx >= 0. && dy >= 0. {
            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(xmin, ymin).of_size(dx as u32, dy as u32),
                image::Rgb([255, 0, 0]),
            );
        }
        for kp in b.data.iter() {
            if kp.mask < 0.6 {
                continue;
            }
            let x = (kp.x * w_ratio) as i32;
            let y = (kp.y * h_ratio) as i32;
            imageproc::drawing::draw_filled_circle_mut(
                &mut img,
                (x, y),
                2,
                image::Rgb([0, 255, 0]),
            );
        }

        for &(idx1, idx2) in KP_CONNECTIONS.iter() {
            let kp1 = &b.data[idx1];
            let kp2 = &b.data[idx2];
            if kp1.mask < 0.6 || kp2.mask < 0.6 {
                continue;
            }
            imageproc::drawing::draw_line_segment_mut(
                &mut img,
                (kp1.x * w_ratio, kp1.y * h_ratio),
                (kp2.x * w_ratio, kp2.y * h_ratio),
                image::Rgb([255, 255, 0]),
            );
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum Which {
    N,
    S,
    M,
    L,
    X,
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum YoloTask {
    Detect,
    Pose,
}

pub trait Task: Module + Sized {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self>;
    fn report(
        pred: &Tensor,
        img: DynamicImage,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> Result<DynamicImage>;
}

impl Task for YoloV8 {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self> {
        println!("XXXXXXXXXXXXXXXX");
        YoloV8::load(vb, multiples, /* num_classes=*/ 80)
    }

    fn report(
        pred: &Tensor,
        img: DynamicImage,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        legend_size: u32,
    ) -> Result<DynamicImage> {
        report_detect(
            pred,
            img,
            w,
            h,
            confidence_threshold,
            nms_threshold,
            legend_size,
        )
    }
}

impl Task for YoloV8Pose {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self> {
        YoloV8Pose::load(vb, multiples, /* num_classes=*/ 1, (17, 3))
    }

    fn report(
        pred: &Tensor,
        img: DynamicImage,
        w: usize,
        h: usize,
        confidence_threshold: f32,
        nms_threshold: f32,
        _legend_size: u32,
    ) -> Result<DynamicImage> {
        report_pose(pred, img, w, h, confidence_threshold, nms_threshold)
    }
}

pub fn run<T: Task>(args: Args) -> anyhow::Result<()> {
    let device = device(args.cpu)?;
    // Create the model and load the weights from the file.
    let multiples = match args.which {
        Which::N => Multiples::n(),
        Which::S => Multiples::s(),
        Which::M => Multiples::m(),
        Which::L => Multiples::l(),
        Which::X => Multiples::x(),
    };
    let model = args.model()?;
println!("MODEL {}", model.display());
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    let model = T::load(vb, multiples)?;
    println!("model loaded");
    for image_name in args.images.iter() {
        println!("processing {image_name}");
        let mut image_name = std::path::PathBuf::from(image_name);
        let original_image = image::io::Reader::open(&image_name)?
            .decode()
            .map_err(candle_core::Error::wrap)?;
        let (width, height) = {
            let w = original_image.width() as usize;
            let h = original_image.height() as usize;
            if w < h {
                let w = w * 640 / h;
                // Sizes have to be divisible by 32.
                (w / 32 * 32, 640)
            } else {
                let h = h * 640 / w;
                (640, h / 32 * 32)
            }
        };
        let image_t = {
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let data = img.to_rgb8().into_raw();
            Tensor::from_vec(
                data,
                (img.height() as usize, img.width() as usize, 3),
                &device,
            )?
            .permute((2, 0, 1))?
        };
        let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = model.forward(&image_t)?.squeeze(0)?;
        println!("generated predictions {predictions:?}");
        let image_t = T::report(
            &predictions,
            original_image,
            width,
            height,
            args.confidence_threshold,
            args.nms_threshold,
            args.legend_size,
        )?;
        image_name.set_extension("pp.jpg");
        println!("writing {image_name:?}");
        image_t.save(image_name)?
    }

    Ok(())
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Multiples {
    depth: f64,
    width: f64,
    ratio: f64,
}

impl Multiples {
    pub fn n() -> Self {
        Self {
            depth: 0.33,
            width: 0.25,
            ratio: 2.0,
        }
    }
    pub fn s() -> Self {
        Self {
            depth: 0.33,
            width: 0.50,
            ratio: 2.0,
        }
    }
    pub fn m() -> Self {
        Self {
            depth: 0.67,
            width: 0.75,
            ratio: 1.5,
        }
    }
    pub fn l() -> Self {
        Self {
            depth: 1.00,
            width: 1.00,
            ratio: 1.0,
        }
    }
    pub fn x() -> Self {
        Self {
            depth: 1.00,
            width: 1.25,
            ratio: 1.0,
        }
    }

    pub fn filters(&self) -> (usize, usize, usize) {
        let f1 = (256. * self.width) as usize;
        let f2 = (512. * self.width) as usize;
        let f3 = (512. * self.width * self.ratio) as usize;
        (f1, f2, f3)
    }
}

#[derive(Debug)]
struct Upsample {
    scale_factor: usize,
}

impl Upsample {
    fn new(scale_factor: usize) -> Result<Self> {
        Ok(Upsample { scale_factor })
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (_b_size, _channels, h, w) = xs.dims4()?;
        xs.upsample_nearest2d(self.scale_factor * h, self.scale_factor * w)
    }
}

#[derive(Debug)]
pub struct ConvBlock {
    conv: Conv2d,
    span: tracing::Span,
}

impl ConvBlock {
    fn load(
        vb: VarBuilder,
        c1: usize,
        c2: usize,
        k: usize,
        stride: usize,
        padding: Option<usize>,
    ) -> Result<Self> {
        let padding = padding.unwrap_or(k / 2);
        let cfg = Conv2dConfig {
            padding,
            stride,
            groups: 1,
            dilation: 1,
        };
        let bn = batch_norm(c2, 1e-3, vb.pp("bn"))?;
        let conv = conv2d_no_bias(c1, c2, k, cfg, vb.pp("conv"))?.absorb_bn(&bn)?;
        Ok(Self {
            conv,
            span: tracing::span!(tracing::Level::TRACE, "conv-block"),
        })
    }
}

impl Module for ConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.conv.forward(xs)?;
        candle_nn::ops::silu(&xs)
    }
}

#[derive(Debug)]
struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    residual: bool,
    span: tracing::Span,
}

impl Bottleneck {
    fn load(vb: VarBuilder, c1: usize, c2: usize, shortcut: bool) -> Result<Self> {
        let channel_factor = 1.;
        let c_ = (c2 as f64 * channel_factor) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, c_, 3, 1, None)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_, c2, 3, 1, None)?;
        let residual = c1 == c2 && shortcut;
        Ok(Self {
            cv1,
            cv2,
            residual,
            span: tracing::span!(tracing::Level::TRACE, "bottleneck"),
        })
    }
}

impl Module for Bottleneck {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let ys = self.cv2.forward(&self.cv1.forward(xs)?)?;
        if self.residual {
            xs + ys
        } else {
            Ok(ys)
        }
    }
}

#[derive(Debug)]
struct C2f {
    cv1: ConvBlock,
    cv2: ConvBlock,
    bottleneck: Vec<Bottleneck>,
    span: tracing::Span,
}

impl C2f {
    fn load(vb: VarBuilder, c1: usize, c2: usize, n: usize, shortcut: bool) -> Result<Self> {
        let c = (c2 as f64 * 0.5) as usize;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, 2 * c, 1, 1, None)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), (2 + n) * c, c2, 1, 1, None)?;
        let mut bottleneck = Vec::with_capacity(n);
        for idx in 0..n {
            let b = Bottleneck::load(vb.pp(&format!("bottleneck.{idx}")), c, c, shortcut)?;
            bottleneck.push(b)
        }
        Ok(Self {
            cv1,
            cv2,
            bottleneck,
            span: tracing::span!(tracing::Level::TRACE, "c2f"),
        })
    }
}

impl Module for C2f {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let ys = self.cv1.forward(xs)?;
        let mut ys = ys.chunk(2, 1)?;
        for m in self.bottleneck.iter() {
            ys.push(m.forward(ys.last().unwrap())?)
        }
        let zs = Tensor::cat(ys.as_slice(), 1)?;
        self.cv2.forward(&zs)
    }
}

#[derive(Debug)]
struct Sppf {
    cv1: ConvBlock,
    cv2: ConvBlock,
    k: usize,
    span: tracing::Span,
}

impl Sppf {
    fn load(vb: VarBuilder, c1: usize, c2: usize, k: usize) -> Result<Self> {
        let c_ = c1 / 2;
        let cv1 = ConvBlock::load(vb.pp("cv1"), c1, c_, 1, 1, None)?;
        let cv2 = ConvBlock::load(vb.pp("cv2"), c_ * 4, c2, 1, 1, None)?;
        Ok(Self {
            cv1,
            cv2,
            k,
            span: tracing::span!(tracing::Level::TRACE, "sppf"),
        })
    }
}

impl Module for Sppf {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_, _, _, _) = xs.dims4()?;
        let xs = self.cv1.forward(xs)?;
        let xs2 = xs
            .pad_with_zeros(2, self.k / 2, self.k / 2)?
            .pad_with_zeros(3, self.k / 2, self.k / 2)?
            .max_pool2d_with_stride(self.k, 1)?;
        let xs3 = xs2
            .pad_with_zeros(2, self.k / 2, self.k / 2)?
            .pad_with_zeros(3, self.k / 2, self.k / 2)?
            .max_pool2d_with_stride(self.k, 1)?;
        let xs4 = xs3
            .pad_with_zeros(2, self.k / 2, self.k / 2)?
            .pad_with_zeros(3, self.k / 2, self.k / 2)?
            .max_pool2d_with_stride(self.k, 1)?;
        self.cv2.forward(&Tensor::cat(&[&xs, &xs2, &xs3, &xs4], 1)?)
    }
}

#[derive(Debug)]
struct Dfl {
    conv: Conv2d,
    num_classes: usize,
    span: tracing::Span,
}

impl Dfl {
    fn load(vb: VarBuilder, num_classes: usize) -> Result<Self> {
        let conv = conv2d_no_bias(num_classes, 1, 1, Default::default(), vb.pp("conv"))?;
        Ok(Self {
            conv,
            num_classes,
            span: tracing::span!(tracing::Level::TRACE, "dfl"),
        })
    }
}

impl Module for Dfl {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, _channels, anchors) = xs.dims3()?;
        let xs = xs
            .reshape((b_sz, 4, self.num_classes, anchors))?
            .transpose(2, 1)?;
        let xs = candle_nn::ops::softmax(&xs, 1)?;
        self.conv.forward(&xs)?.reshape((b_sz, 4, anchors))
    }
}

#[derive(Debug)]
pub struct DarkNet {
    b1_0: ConvBlock,
    b1_1: ConvBlock,
    b2_0: C2f,
    b2_1: ConvBlock,
    b2_2: C2f,
    b3_0: ConvBlock,
    b3_1: C2f,
    b4_0: ConvBlock,
    b4_1: C2f,
    b5: Sppf,
    span: tracing::Span,
}

impl DarkNet {
    pub fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let (w, r, d) = (m.width, m.ratio, m.depth);
        let b1_0 = ConvBlock::load(vb.pp("b1.0"), 3, (64. * w) as usize, 3, 2, Some(1))?;
        let b1_1 = ConvBlock::load(
            vb.pp("b1.1"),
            (64. * w) as usize,
            (128. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let b2_0 = C2f::load(
            vb.pp("b2.0"),
            (128. * w) as usize,
            (128. * w) as usize,
            (3. * d).round() as usize,
            true,
        )?;
        let b2_1 = ConvBlock::load(
            vb.pp("b2.1"),
            (128. * w) as usize,
            (256. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let b2_2 = C2f::load(
            vb.pp("b2.2"),
            (256. * w) as usize,
            (256. * w) as usize,
            (6. * d).round() as usize,
            true,
        )?;
        let b3_0 = ConvBlock::load(
            vb.pp("b3.0"),
            (256. * w) as usize,
            (512. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let b3_1 = C2f::load(
            vb.pp("b3.1"),
            (512. * w) as usize,
            (512. * w) as usize,
            (6. * d).round() as usize,
            true,
        )?;
        let b4_0 = ConvBlock::load(
            vb.pp("b4.0"),
            (512. * w) as usize,
            (512. * w * r) as usize,
            3,
            2,
            Some(1),
        )?;
        let b4_1 = C2f::load(
            vb.pp("b4.1"),
            (512. * w * r) as usize,
            (512. * w * r) as usize,
            (3. * d).round() as usize,
            true,
        )?;
        let b5 = Sppf::load(
            vb.pp("b5.0"),
            (512. * w * r) as usize,
            (512. * w * r) as usize,
            5,
        )?;
        Ok(Self {
            b1_0,
            b1_1,
            b2_0,
            b2_1,
            b2_2,
            b3_0,
            b3_1,
            b4_0,
            b4_1,
            b5,
            span: tracing::span!(tracing::Level::TRACE, "darknet"),
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let _enter = self.span.enter();
        let x1 = self.b1_1.forward(&self.b1_0.forward(xs)?)?;
        let x2 = self
            .b2_2
            .forward(&self.b2_1.forward(&self.b2_0.forward(&x1)?)?)?;
        let x3 = self.b3_1.forward(&self.b3_0.forward(&x2)?)?;
        let x4 = self.b4_1.forward(&self.b4_0.forward(&x3)?)?;
        let x5 = self.b5.forward(&x4)?;
        Ok((x2, x3, x5))
    }
}

#[derive(Debug)]
pub struct YoloV8Neck {
    up: Upsample,
    n1: C2f,
    n2: C2f,
    n3: ConvBlock,
    n4: C2f,
    n5: ConvBlock,
    n6: C2f,
    span: tracing::Span,
}

impl YoloV8Neck {
    pub fn load(vb: VarBuilder, m: Multiples) -> Result<Self> {
        let up = Upsample::new(2)?;
        let (w, r, d) = (m.width, m.ratio, m.depth);
        let n = (3. * d).round() as usize;
        let n1 = C2f::load(
            vb.pp("n1"),
            (512. * w * (1. + r)) as usize,
            (512. * w) as usize,
            n,
            false,
        )?;
        let n2 = C2f::load(
            vb.pp("n2"),
            (768. * w) as usize,
            (256. * w) as usize,
            n,
            false,
        )?;
        let n3 = ConvBlock::load(
            vb.pp("n3"),
            (256. * w) as usize,
            (256. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let n4 = C2f::load(
            vb.pp("n4"),
            (768. * w) as usize,
            (512. * w) as usize,
            n,
            false,
        )?;
        let n5 = ConvBlock::load(
            vb.pp("n5"),
            (512. * w) as usize,
            (512. * w) as usize,
            3,
            2,
            Some(1),
        )?;
        let n6 = C2f::load(
            vb.pp("n6"),
            (512. * w * (1. + r)) as usize,
            (512. * w * r) as usize,
            n,
            false,
        )?;
        Ok(Self {
            up,
            n1,
            n2,
            n3,
            n4,
            n5,
            n6,
            span: tracing::span!(tracing::Level::TRACE, "neck"),
        })
    }

    pub fn forward(&self, p3: &Tensor, p4: &Tensor, p5: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let _enter = self.span.enter();
        let x = self
            .n1
            .forward(&Tensor::cat(&[&self.up.forward(p5)?, p4], 1)?)?;
        let head_1 = self
            .n2
            .forward(&Tensor::cat(&[&self.up.forward(&x)?, p3], 1)?)?;
        let head_2 = self
            .n4
            .forward(&Tensor::cat(&[&self.n3.forward(&head_1)?, &x], 1)?)?;
        let head_3 = self
            .n6
            .forward(&Tensor::cat(&[&self.n5.forward(&head_2)?, p5], 1)?)?;
        Ok((head_1, head_2, head_3))
    }
}

#[derive(Debug)]
pub struct DetectionHead {
    dfl: Dfl,
    cv2: [(ConvBlock, ConvBlock, Conv2d); 3],
    cv3: [(ConvBlock, ConvBlock, Conv2d); 3],
    ch: usize,
    no: usize,
    span: tracing::Span,
}

#[derive(Debug)]
struct PoseHead {
    detect: DetectionHead,
    cv4: [(ConvBlock, ConvBlock, Conv2d); 3],
    kpt: (usize, usize),
    span: tracing::Span,
}

fn make_anchors(
    xs0: &Tensor,
    xs1: &Tensor,
    xs2: &Tensor,
    (s0, s1, s2): (usize, usize, usize),
    grid_cell_offset: f64,
) -> Result<(Tensor, Tensor)> {
    let dev = xs0.device();
    let mut anchor_points = vec![];
    let mut stride_tensor = vec![];
    for (xs, stride) in [(xs0, s0), (xs1, s1), (xs2, s2)] {
        // xs is only used to extract the h and w dimensions.
        let (_, _, h, w) = xs.dims4()?;
        let sx = (Tensor::arange(0, w as u32, dev)?.to_dtype(DType::F32)? + grid_cell_offset)?;
        let sy = (Tensor::arange(0, h as u32, dev)?.to_dtype(DType::F32)? + grid_cell_offset)?;
        let sx = sx
            .reshape((1, sx.elem_count()))?
            .repeat((h, 1))?
            .flatten_all()?;
        let sy = sy
            .reshape((sy.elem_count(), 1))?
            .repeat((1, w))?
            .flatten_all()?;
        anchor_points.push(Tensor::stack(&[&sx, &sy], D::Minus1)?);
        stride_tensor.push((Tensor::ones(h * w, DType::F32, dev)? * stride as f64)?);
    }
    let anchor_points = Tensor::cat(anchor_points.as_slice(), 0)?;
    let stride_tensor = Tensor::cat(stride_tensor.as_slice(), 0)?.unsqueeze(1)?;
    Ok((anchor_points, stride_tensor))
}
fn dist2bbox(distance: &Tensor, anchor_points: &Tensor) -> Result<Tensor> {
    let chunks = distance.chunk(2, 1)?;
    let lt = &chunks[0];
    let rb = &chunks[1];
    let x1y1 = anchor_points.sub(lt)?;
    let x2y2 = anchor_points.add(rb)?;
    let c_xy = ((&x1y1 + &x2y2)? * 0.5)?;
    let wh = (&x2y2 - &x1y1)?;
    Tensor::cat(&[c_xy, wh], 1)
}

pub struct DetectionHeadOut {
    pred: Tensor,
    anchors: Tensor,
    strides: Tensor,
}

impl DetectionHead {
    pub fn load(vb: VarBuilder, nc: usize, filters: (usize, usize, usize)) -> Result<Self> {
        let ch = 16;
        let dfl = Dfl::load(vb.pp("dfl"), ch)?;
        let c1 = usize::max(filters.0, nc);
        let c2 = usize::max(filters.0 / 4, ch * 4);
        let cv3 = [
            Self::load_cv3(vb.pp("cv3.0"), c1, nc, filters.0)?,
            Self::load_cv3(vb.pp("cv3.1"), c1, nc, filters.1)?,
            Self::load_cv3(vb.pp("cv3.2"), c1, nc, filters.2)?,
        ];
        let cv2 = [
            Self::load_cv2(vb.pp("cv2.0"), c2, ch, filters.0)?,
            Self::load_cv2(vb.pp("cv2.1"), c2, ch, filters.1)?,
            Self::load_cv2(vb.pp("cv2.2"), c2, ch, filters.2)?,
        ];
        let no = nc + ch * 4;
        Ok(Self {
            dfl,
            cv2,
            cv3,
            ch,
            no,
            span: tracing::span!(tracing::Level::TRACE, "detection-head"),
        })
    }

    pub fn load_cv3(
        vb: VarBuilder,
        c1: usize,
        nc: usize,
        filter: usize,
    ) -> Result<(ConvBlock, ConvBlock, Conv2d)> {
        let block0 = ConvBlock::load(vb.pp("0"), filter, c1, 3, 1, None)?;
        let block1 = ConvBlock::load(vb.pp("1"), c1, c1, 3, 1, None)?;
        let conv = conv2d(c1, nc, 1, Default::default(), vb.pp("2"))?;
        Ok((block0, block1, conv))
    }

    pub fn load_cv2(
        vb: VarBuilder,
        c2: usize,
        ch: usize,
        filter: usize,
    ) -> Result<(ConvBlock, ConvBlock, Conv2d)> {
        let block0 = ConvBlock::load(vb.pp("0"), filter, c2, 3, 1, None)?;
        let block1 = ConvBlock::load(vb.pp("1"), c2, c2, 3, 1, None)?;
        let conv = conv2d(c2, 4 * ch, 1, Default::default(), vb.pp("2"))?;
        Ok((block0, block1, conv))
    }

    pub fn forward(&self, xs0: &Tensor, xs1: &Tensor, xs2: &Tensor) -> Result<DetectionHeadOut> {
        let _enter = self.span.enter();
        let forward_cv = |xs, i: usize| {
            let xs_2 = self.cv2[i].0.forward(xs)?;
            let xs_2 = self.cv2[i].1.forward(&xs_2)?;
            let xs_2 = self.cv2[i].2.forward(&xs_2)?;

            let xs_3 = self.cv3[i].0.forward(xs)?;
            let xs_3 = self.cv3[i].1.forward(&xs_3)?;
            let xs_3 = self.cv3[i].2.forward(&xs_3)?;
            Tensor::cat(&[&xs_2, &xs_3], 1)
        };
        let xs0 = forward_cv(xs0, 0)?;
        let xs1 = forward_cv(xs1, 1)?;
        let xs2 = forward_cv(xs2, 2)?;

        let (anchors, strides) = make_anchors(&xs0, &xs1, &xs2, (8, 16, 32), 0.5)?;
        let anchors = anchors.transpose(0, 1)?.unsqueeze(0)?;
        let strides = strides.transpose(0, 1)?;

        let reshape = |xs: &Tensor| {
            let d = xs.dim(0)?;
            let el = xs.elem_count();
            xs.reshape((d, self.no, el / (d * self.no)))
        };
        let ys0 = reshape(&xs0)?;
        let ys1 = reshape(&xs1)?;
        let ys2 = reshape(&xs2)?;

        let x_cat = Tensor::cat(&[ys0, ys1, ys2], 2)?;
        let box_ = x_cat.i((.., ..self.ch * 4))?;
        let cls = x_cat.i((.., self.ch * 4..))?;

        let dbox = dist2bbox(&self.dfl.forward(&box_)?, &anchors)?;
        let dbox = dbox.broadcast_mul(&strides)?;
        let pred = Tensor::cat(&[dbox, candle_nn::ops::sigmoid(&cls)?], 1)?;
        Ok(DetectionHeadOut {
            pred,
            anchors,
            strides,
        })
    }
}

impl PoseHead {
    // kpt: keypoints, (17, 3)
    // nc: num-classes, 80
    fn load(
        vb: VarBuilder,
        nc: usize,
        kpt: (usize, usize),
        filters: (usize, usize, usize),
    ) -> Result<Self> {
        let detect = DetectionHead::load(vb.clone(), nc, filters)?;
        let nk = kpt.0 * kpt.1;
        let c4 = usize::max(filters.0 / 4, nk);
        let cv4 = [
            Self::load_cv4(vb.pp("cv4.0"), c4, nk, filters.0)?,
            Self::load_cv4(vb.pp("cv4.1"), c4, nk, filters.1)?,
            Self::load_cv4(vb.pp("cv4.2"), c4, nk, filters.2)?,
        ];
        Ok(Self {
            detect,
            cv4,
            kpt,
            span: tracing::span!(tracing::Level::TRACE, "pose-head"),
        })
    }

    fn load_cv4(
        vb: VarBuilder,
        c1: usize,
        nc: usize,
        filter: usize,
    ) -> Result<(ConvBlock, ConvBlock, Conv2d)> {
        let block0 = ConvBlock::load(vb.pp("0"), filter, c1, 3, 1, None)?;
        let block1 = ConvBlock::load(vb.pp("1"), c1, c1, 3, 1, None)?;
        let conv = conv2d(c1, nc, 1, Default::default(), vb.pp("2"))?;
        Ok((block0, block1, conv))
    }

    fn forward(&self, xs0: &Tensor, xs1: &Tensor, xs2: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let d = self.detect.forward(xs0, xs1, xs2)?;
        let forward_cv = |xs: &Tensor, i: usize| {
            let (b_sz, _, h, w) = xs.dims4()?;
            let xs = self.cv4[i].0.forward(xs)?;
            let xs = self.cv4[i].1.forward(&xs)?;
            let xs = self.cv4[i].2.forward(&xs)?;
            xs.reshape((b_sz, self.kpt.0 * self.kpt.1, h * w))
        };
        let xs0 = forward_cv(xs0, 0)?;
        let xs1 = forward_cv(xs1, 1)?;
        let xs2 = forward_cv(xs2, 2)?;
        let xs = Tensor::cat(&[xs0, xs1, xs2], D::Minus1)?;
        let (b_sz, _nk, hw) = xs.dims3()?;
        let xs = xs.reshape((b_sz, self.kpt.0, self.kpt.1, hw))?;

        let ys01 = ((xs.i((.., .., 0..2))? * 2.)?.broadcast_add(&d.anchors)? - 0.5)?
            .broadcast_mul(&d.strides)?;
        let ys2 = candle_nn::ops::sigmoid(&xs.i((.., .., 2..3))?)?;
        let ys = Tensor::cat(&[ys01, ys2], 2)?.flatten(1, 2)?;
        Tensor::cat(&[d.pred, ys], 1)
    }
}

#[derive(Debug)]
pub struct YoloV8 {
    pub net: DarkNet,
    pub fpn: YoloV8Neck,
    pub head: DetectionHead,
    pub span: tracing::Span,
}

impl YoloV8 {
    pub fn load(vb: VarBuilder, m: Multiples, num_classes: usize) -> Result<Self> {
        let net = DarkNet::load(vb.pp("net"), m)?;
        let fpn = YoloV8Neck::load(vb.pp("fpn"), m)?;
        let head = DetectionHead::load(vb.pp("head"), num_classes, m.filters())?;
        Ok(Self {
            net,
            fpn,
            head,
            span: tracing::span!(tracing::Level::TRACE, "yolo-v8"),
        })
    }
}

impl Module for YoloV8 {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (xs1, xs2, xs3) = self.net.forward(xs)?;
        let (xs1, xs2, xs3) = self.fpn.forward(&xs1, &xs2, &xs3)?;
        Ok(self.head.forward(&xs1, &xs2, &xs3)?.pred)
    }
}

#[derive(Debug)]
pub struct YoloV8Pose {
    net: DarkNet,
    fpn: YoloV8Neck,
    head: PoseHead,
    span: tracing::Span,
}

impl YoloV8Pose {
    pub fn load(
        vb: VarBuilder,
        m: Multiples,
        num_classes: usize,
        kpt: (usize, usize),
    ) -> Result<Self> {
        let net = DarkNet::load(vb.pp("net"), m)?;
        let fpn = YoloV8Neck::load(vb.pp("fpn"), m)?;
        let head = PoseHead::load(vb.pp("head"), num_classes, kpt, m.filters())?;
        Ok(Self {
            net,
            fpn,
            head,
            span: tracing::span!(tracing::Level::TRACE, "yolo-v8-pose"),
        })
    }
}

impl Module for YoloV8Pose {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (xs1, xs2, xs3) = self.net.forward(xs)?;
        let (xs1, xs2, xs3) = self.fpn.forward(&xs1, &xs2, &xs3)?;
        self.head.forward(&xs1, &xs2, &xs3)
    }
}

