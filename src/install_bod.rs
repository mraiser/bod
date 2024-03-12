use std::path::PathBuf;

pub fn install_bod(bod_path: PathBuf) {
  if !bod_path.exists() {
    let _x = std::fs::create_dir_all(&bod_path);
  }
  
  let f = bod_path.join("yolov8s.safetensors");
  if !f.exists() {
    let b = Vec::from(include_bytes!("yolov8s.safetensors") as &[u8]);
    std::fs::write(f, &b).expect("Unable to write file");
  }
  
  let f = bod_path.join("shape_predictor_68_face_landmarks.dat");
  if !f.exists() {
    let b = Vec::from(include_bytes!("shape_predictor_68_face_landmarks.dat") as &[u8]);
    std::fs::write(f, &b).expect("Unable to write file");
  }
  
  let f = bod_path.join("dlib_face_recognition_resnet_model_v1.dat");
  if !f.exists() {
    let b = Vec::from(include_bytes!("dlib_face_recognition_resnet_model_v1.dat") as &[u8]);
    std::fs::write(f, &b).expect("Unable to write file");
  }
}

