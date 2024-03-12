mod yo;
mod install_bod;

use yo::*;
use std::env;
use crate::install_bod::install_bod;

use candle_core::Device;
use candle_nn::VarBuilder;
use candle_core::DType;
use candle_core::Tensor;
use candle_core::Module;
use candle_core::IndexOp;
use ndata::dataarray::DataArray;
use std::path::PathBuf;
use image::imageops::crop_imm;
use image::GenericImageView;
use dlib_face_recognition::FaceDetector;
use dlib_face_recognition::ImageMatrix;
use dlib_face_recognition::FaceDetectorTrait;
use dlib_face_recognition::LandmarkPredictor;
use dlib_face_recognition::FaceEncoderNetwork;
use dlib_face_recognition::LandmarkPredictorTrait;
use dlib_face_recognition::FaceEncoderTrait;
use dlib_face_recognition::FaceEncoding;
use ndata::databytes::DataBytes;
use ndata::dataobject::DataObject;
use std::path::Path;
use core::time::Duration;
use image::ImageFormat;

fn main() {
  let _q = ndata::init();
  let args: Vec<String> = env::args().collect();
  
  let mut b = false;
  if args.len() == 5 {
    let workdir = Path::new(&args[1]).canonicalize();
    if workdir.is_ok() {
      let threshold = (&args[2]).parse::<f64>();
      if threshold.is_ok() {
        let minside = (&args[3]).parse::<u32>();
        if minside.is_ok() {
          let numthreads = (&args[4]).parse::<u8>();
          if numthreads.is_ok() {
            let workdir = workdir.unwrap();

            let deps = workdir.join("bin");
            install_bod(deps);

            let a0 = workdir.display().to_string();
            let a1 = threshold.unwrap();
            let a2:i64 = minside.unwrap().into();
            let a3:i64 = numthreads.unwrap().into();
            let ax = crop_raw(a0, a1, a2, a3);
            println!("{}", ax.to_string());
          
            b = true;
          }
        }
      }
    }
  }
  
  if !b {
    println!("USAGE: bod WORKDIR 0.6 256 6");
    println!("where WORKDIR is a directory that contains a directory named raw containing png images");
    println!("and 0.6 is the desired similarity threshold (0.0 is identical)");
    println!("and 256 is the desired minimum size in pixels of at least one side");
    println!("and 6 is the desired number of threads");
  }

//  let img = &args[1];
}

pub fn crop_raw(workdir:String, threshold:f64, min_side:i64, num_threads:i64) -> DataObject {
  let workdir = Path::new(&workdir);
  let image_path = workdir.join("raw");
  let dest_dir = workdir.join("people");
  let bin_dir = workdir.join("bin");
  let _x = std::fs::create_dir_all(&dest_dir);

  let groups = DataArray::new();
  let mut stack = DataArray::new();
  let locks = DataArray::new();

  let paths = std::fs::read_dir(image_path.clone()).unwrap();
  for path in paths {
    let path = path.unwrap().path();
    stack.push_string(&path.display().to_string());
  }

  for n in 0..num_threads {
    let mut stack = stack.clone();
    let mut locks = locks.clone();
    let groups = groups.clone();
    let dest_dir = dest_dir.clone();
    let bin_dir = bin_dir.clone();
    locks.push_null();

    std::thread::spawn(move || {
      println!("START THREAD {} CROP", n);

      let tools = new_tools(bin_dir);

      while stack.len() > 0 {
        let fullname = stack.pop_property(0).string();
        let path = Path::new(&fullname);
        //let filename = path.clone().file_name().unwrap().to_os_string().into_string().unwrap();
        let _list = process_file(path.to_path_buf(), dest_dir.to_path_buf(), &tools, groups.clone(), min_side, threshold);
      }
      println!("END THREAD {} CROP", n);
      locks.remove_property(0);
    });
  }

  let beat = Duration::from_millis(100);
  while locks.len() > 0 {
    std::thread::sleep(beat);
  }

  println!("DONE CROPPING AND SORTING");

  DataObject::new()
}

pub struct Toolset {
  model:YoloV8,
  device:Device,
  detector:FaceDetector,
  landmarks:LandmarkPredictor,
  face_encoder:FaceEncoderNetwork
}

pub fn new_tools(bod_path: PathBuf) -> Toolset {
  let device = device(false).unwrap(); 
  let model = bod_path.join("yolov8s.safetensors");
  let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device).unwrap() };
  let m = Multiples::s();
  let num_classes = 80; // FIXME - we only need 0 for person
  let net = DarkNet::load(vb.pp("net"), m).unwrap();
  let fpn = YoloV8Neck::load(vb.pp("fpn"), m).unwrap();
  let head = DetectionHead::load(vb.pp("head"), num_classes, m.filters()).unwrap();

  let model = YoloV8 {
      net: net,
      fpn: fpn,
      head: head,
      span: tracing::span!(tracing::Level::TRACE, "yolo-v8"),
  };

  let detector = FaceDetector::default();
  let landmarks = LandmarkPredictor::open(bod_path.join("shape_predictor_68_face_landmarks.dat")).unwrap();
  let face_encoder = FaceEncoderNetwork::open(bod_path.join("dlib_face_recognition_resnet_model_v1.dat")).unwrap();
  
  Toolset {
    model:model,
    device:device,
    detector:detector,
    landmarks:landmarks,
    face_encoder:face_encoder
  }
}

pub fn process_file(image_path:PathBuf, dest_dir:PathBuf, tools:&Toolset, mut groups:DataArray, min_side:i64, threshold:f64) -> DataArray {
  let model = &tools.model;
  let device = &tools.device;
  let detector = &tools.detector;
  let landmarks = &tools.landmarks;
  let face_encoder = &tools.face_encoder;

  let confidence_threshold = 0.25;
  let mut list = DataArray::new();
  let body_padding: [f64; 4] = [0.1,0.1,0.1,0.1];
  let face_padding: [f64; 4] = [0.1,0.33,0.1,0.1];
  let mut index = 0;
  
  let filename = image_path.file_name().unwrap().to_os_string().into_string().unwrap();
  let original_image = image::io::Reader::open(&image_path).unwrap().decode();
  if original_image.is_ok() {
    let original_image = original_image.unwrap();
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
        let mut bod = DataArray::new();
        bod.push_int(xmin);
        bod.push_int(ymin);
        bod.push_int(dx);
        bod.push_int(dy);
        
        
        let x = xmin as u32;
        let y = ymin as u32;
        let w = dx as u32;
        let h = dy as u32;
        //println!("check1 {}, {}, {}, {}", x, y, w, h);
        
        let imgw = initial_w;
        let imgh = initial_h;
        let image = original_image.to_rgb8();
        
        
        
        let a = pad([x, y, x+w, y+h], imgw, imgh, body_padding);
        let c = crop_imm(&image, a[0], a[1], a[2], a[3]);
        let cw = c.width() as i64;
        let ch = c.height() as i64;
        
        if (cw > 0 && ch > 0) && (cw >= min_side || ch >= min_side) {
          let c = c.to_image();
          
          
          
          let matrix = ImageMatrix::from_image(&c);
          let face_locations = detector.face_locations(&matrix);
          if face_locations.len() > 0 {
            let r = face_locations[0];
            let landmarks = landmarks.face_landmarks(&matrix, &r);
            let encodings = face_encoder.get_face_encodings(&matrix, &[landmarks], 0);
            if encodings.len() > 0 {
              let enc1 = encodings[0].clone();
              
              let mut j = 0;
              let mut found = false;
              for enc2 in groups.clone().objects() {
                let enc2 = enc2.bytes();
                let enc2 = enc2.get_data();
                let q;
                unsafe { q = std::slice::from_raw_parts(enc2.as_ptr() as *const f64, enc2.len() / 8).to_owned(); }
                let enc2 = FaceEncoding::from_vec(&q.to_vec()).unwrap();
                if enc1.distance(&enc2) <= threshold {
                  found = true;
                  //println!("join {}", j);
                  break;
                }
                j += 1;
              }
              if !found {
                let b:&[f64] = enc1.as_ref();
                let xxx;
                unsafe { xxx = std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 8); }
                groups.push_bytes(DataBytes::from_bytes(&xxx.to_vec()));
                //println!("form {}", j);
              }
              
              let filenamex = filename.to_string()+"-"+(&index.to_string())+"-BODY.png";
              let cfile = dest_dir.join(&j.to_string());
              let _x = std::fs::create_dir_all(&cfile);
              let cfile = cfile.join(&filenamex);
              let _ = c.save_with_format(&cfile, ImageFormat::Png).unwrap();
              
              let mut x = r.left as u32;
              let mut y = r.top as u32;
              let mut x2 = r.right as u32;
              let mut y2 = r.bottom as u32;
              if x > x2 { let x3 = x; x = x2; x2 = x3; }
              if y > y2 { let y3 = y; y = y2; y2 = y3; }
              let w = x2 - x;
              let h = y2 - y;
              //println!("check1 {}, {}, {}, {}, {}, {}", x, y, w, h, cw, ch);
              let a = pad([x, y, x+w, y+h], cw as u32, ch as u32, face_padding);
              let c = crop_imm(&c, a[0], a[1], a[2], a[3]);// FIXME - expensive, pointless, and lazy
              let cw = c.width() as i64;
              let ch = c.height() as i64;

              let mut face = DataArray::new();
              face.push_int(a[0] as i64);
              face.push_int(a[1] as i64);
              face.push_int(cw);
              face.push_int(ch);
              
              if (cw > 0 && ch > 0) && (cw >= min_side || ch >= min_side) {
                let c = c.to_image();
              
                let filenamex = filename.to_string()+"-"+(&index.to_string())+"-FACE.png";
                let cfile = dest_dir.join(&j.to_string());
                let _x = std::fs::create_dir_all(&cfile);
                let cfile = cfile.join(&filenamex);
                let _ = c.save_with_format(&cfile, ImageFormat::Png).unwrap();
              }
              
              let mut meta = DataObject::new();
              meta.put_string("file", &filename);
              meta.put_array("bod", bod);
              meta.put_array("face", face);
              meta.put_int("group", j);
              meta.put_int("index", index);
              
              
              
              
              println!("{}", meta.to_string());
              list.push_object(meta);
              index += 1;
            }
          }
        }
        
        //println!("{}", bod.to_string());
      }
    }  
  }
  list
}

fn pad(a:[u32; 4], w:u32, h:u32, d:[f64; 4]) -> [u32; 4]{
  let foundw = a[2] as i64 - a[0] as i64;
  let foundh = a[3] as i64 - a[1] as i64;

  let mut x = a[0] as i64 - (foundw as f64 * d[0]) as i64;
  let mut y = a[1] as i64 - (foundw as f64 * d[1]) as i64;
  let mut x2 = a[2] as i64 + (foundh as f64 * d[2]) as i64;
  let mut y2 = a[3] as i64 + (foundh as f64 * d[3]) as i64;
  
  if x < 0 { x = 0; }
  if y < 0 { y = 0; }
  if x2 > w as i64 { x2 = w as i64; }
  if y2 > h as i64  { y2 = h as i64; }

  [x as u32, y as u32, (x2-x) as u32, (y2-y) as u32]
}

