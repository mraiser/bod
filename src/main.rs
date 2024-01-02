mod yo;

use yo::*;
use std::env;

use candle_core::Device;
use candle_nn::VarBuilder;
use candle_core::DType;
use candle_core::Tensor;
use candle_core::Module;
use candle_core::IndexOp;
use ndata::dataarray::DataArray;

fn main() {
    let args: Vec<String> = env::args().collect();
    let img = &args[1];
    
    let _q = ndata::init();
  
    let device = Device::new_cuda(0).unwrap(); 
    let model = "/home/mraiser/.cache/huggingface/hub/models--lmz--candle-yolo-v8/snapshots/be388c6fab95ae3035a039070e1b883b9c5a1325/yolov8s.safetensors";
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device).unwrap() };
    let m = Multiples::s();
    let num_classes = 80; // FIXME - we only need 0 for person
    let confidence_threshold = 0.25;
    let net = DarkNet::load(vb.pp("net"), m).unwrap();
    let fpn = YoloV8Neck::load(vb.pp("fpn"), m).unwrap();
    let head = DetectionHead::load(vb.pp("head"), num_classes, m.filters()).unwrap();
  
    let model = YoloV8 {
      net: net,
      fpn: fpn,
      head: head,
      span: tracing::span!(tracing::Level::TRACE, "yolo-v8"),
    };

    //let image_name = "/usb1/train/bitsie_tulloch/a5428471_b-628835397.jpg.png";
    let image_name = std::path::PathBuf::from(img);
    let original_image = image::io::Reader::open(&image_name).unwrap().decode().map_err(candle_core::Error::wrap).unwrap();
    let (initial_h, initial_w) = (original_image.height(), original_image.width());
    let (width, height) = {
      let w = initial_w as usize;
      let h = initial_h as usize;
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
      ).unwrap()
      .permute((2, 0, 1)).unwrap()
    };
    let image_t = (image_t.unsqueeze(0).unwrap().to_dtype(DType::F32).unwrap() * (1. / 255.)).unwrap();
    let pred = model.forward(&image_t).unwrap().squeeze(0).unwrap();

    let pred = pred.to_device(&device).unwrap();
    let (pred_size, npreds) = pred.dims2().unwrap();
    let nclasses = pred_size - 4;
    let mut list = DataArray::new();
    for index in 0..npreds {
      let pred = Vec::<f32>::try_from(pred.i((.., index)).unwrap()).unwrap();
      let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
      if confidence > confidence_threshold {
        let mut class_index = 0;
        for i in 0..nclasses {
          if pred[4 + i] > pred[4 + class_index] {
            class_index = i
          }
        }
        if pred[class_index + 4] > 0. {
          if class_index == 0 {
            let mut a = DataArray::new();
            a.push_float(f64::max(0.0,(pred[0] - pred[2] / 2.) as f64));
            a.push_float(f64::max(0.0,(pred[1] - pred[3] / 2.) as f64));
            a.push_float(f64::max(0.0,(pred[0] + pred[2] / 2.) as f64));
            a.push_float(f64::max(0.0,(pred[1] + pred[3] / 2.) as f64));
            list.push_array(a);
          }
        }
      }
    }

    let mut listolists = DataArray::new();

    while list.len() > 0 {
      let mut nextlist = DataArray::new();
      let next = list.pop_property(0).array();
      nextlist.push_array(next.clone());
      listolists.push_array(nextlist.clone());
      
      let xmin = next.get_float(0);
      let ymin = next.get_float(1);
      let xmax = next.get_float(2);
      let ymax = next.get_float(3);
      
      let n = list.len();
      let mut i = n;
      while i>0 {
        i -= 1;
        let q = list.get_array(i);
        let a = xmin - q.get_float(0);
        let b = ymin - q.get_float(1);
        let c = xmax - q.get_float(2);
        let d = ymax - q.get_float(3);
        let score = (a*a)+(b*b)+(c*c)+(d*d);
        if score < 10000.0 {
          nextlist.push_array(q);
          list.pop_property(i);
        }
      }
    }

    let w_ratio = initial_w as f64 / width as f64;
    let h_ratio = initial_h as f64 / height as f64;

    while listolists.len() > 0 {
      let next = listolists.pop_property(0).array();
      let mut curr = [0.0,0.0,0.0,0.0];
      let mut count = 0;
      for list in next.objects(){
        let list = list.array();
        curr[0] += list.get_float(0);
        curr[1] += list.get_float(1);
        curr[2] += list.get_float(2);
        curr[3] += list.get_float(3);
        count += 1;
      }
      let xmin = (curr[0] * w_ratio / count as f64) as i64;
      let ymin = (curr[1] * h_ratio / count as f64) as i64;
      let xmax = (curr[2] * w_ratio / count as f64) as i64;
      let ymax = (curr[3] * h_ratio / count as f64) as i64;
      let dx = xmax - xmin;
      let dy = ymax - ymin;
      if dx >= 0 && dy >= 0 {
        let mut a = DataArray::new();
        a.push_int(xmin);
        a.push_int(ymin);
        a.push_int(dx);
        a.push_int(dy);
        list.push_array(a);
      }
    }  
  
    println!("{}", list.to_string());
}


