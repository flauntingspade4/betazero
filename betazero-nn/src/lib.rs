// use std::{error::Error, path::Path};

use citron_core::Board;
use tensorflow::FetchToken;
use tensorflow::SavedModelBundle;
// use tensorflow::Graph;

pub fn evaluate_position(_board: &Board) -> (f64, [[[f64; 64]; 8]; 8]) {
    todo!("None of this is done")
}

use std::error::Error;
// use std::fs::File;
// use std::io::Read;/
// use std::path::Path;
use std::result::Result;
// use tensorflow::Code;
use tensorflow::Graph;
// use tensorflow::ImportGraphDefOptions;
// use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
// use tensorflow::Status;
use tensorflow::Tensor;

pub fn load_model() -> Result<(), Box<dyn Error>> {
    /*//Sigmatures declared when we saved the model
    let train_input_parameter_input_name = "training_input";
    let train_input_parameter_target_name = "training_target";
    let pred_input_parameter_name = "inputs";

    //Names of output nodes of the graph, retrieved with the saved_model_cli command
    let train_output_parameter_name = "output_0";
    let pred_output_parameter_name = "output_0";

    //Create some tensors to feed to the model for training, one as input and one as the target value
    //Note: All tensors must be declared before args!
    let input_tensor: Tensor<f32> = Tensor::new(&[1, 2]).with_values(&[1.0, 1.0]).unwrap();
    let target_tensor: Tensor<f32> = Tensor::new(&[1, 1]).with_values(&[2.0]).unwrap();*/

    //Path of the saved model
    let save_dir = "model/";

    //Create a graph
    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, save_dir)
        .expect("Can't load saved model");
    let session = &bundle.session;

    dbg!(bundle.meta_graph_def());

    let call_signature = bundle
        .meta_graph_def()
        .get_signature("serve")
        .expect("Signature 'call' not found in saved_mode.pb");
    let input_info = call_signature.get_input("args_0")?;
    let output_info = call_signature.get_output("output_0")?;
    let input_op = graph.operation_by_name(&input_info.name().name)?.unwrap();
    let output_op = graph.operation_by_name(&output_info.name().name)?.unwrap();

    // Create input variables for our addition
    let mut x = Tensor::new(&[1, 2]);
    x[0] = 3.0f32;
    x[1] = 5.;
    dbg!(&x);
    // let mut y = Tensor::new(&[1]);
    // y[0] = 40i32;

    let mut call_step = SessionRunArgs::new();
    call_step.add_feed(&input_op, 0, &x);
    // call_step.add_feed(&graph.operation_by_name_required("y")?, 0, &y);
    // let z = call_step.request_fetch(&graph.operation_by_name_required("z")?, 0);
    // call_step.add_target(operation);
    let result = call_step.request_fetch(&output_op, 0);
    call_step.add_target(&output_op);
    session.run(&mut call_step)?;

    let result: Tensor<f32> = call_step.fetch(result)?;

    // let z_res: i32 = call_step.fetch(z)?[0];
    dbg!(result);
    Ok(())
}

/*pub fn load_model() -> Result<(), Box<dyn Error>> {
    let filename = "models/addition/saved_model.pb"; // z = x + y
    if !Path::new(filename).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python addition.py' to generate {} \
                     and try again.",
                    filename
                ),
            )
            .unwrap(),
        ));
    }

    // Create input variables for our addition
    let mut x = Tensor::new(&[1]);
    x[0] = 2i32;
    let mut y = Tensor::new(&[1]);
    y[0] = 40i32;

    println!("Loading file");
    // Load the computation graph defined by addition.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let session = Session::new(&SessionOptions::new(), &graph)?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(&graph.operation_by_name_required("x")?, 0, &x);
    args.add_feed(&graph.operation_by_name_required("y")?, 0, &y);
    let z = args.request_fetch(&graph.operation_by_name_required("z")?, 0);
    session.run(&mut args)?;

    // Check our results.
    let z_res: i32 = args.fetch(z)?[0];
    println!("{:?}", z_res);

    Ok(())
}
*/
#[test]
fn main_test() -> Result<(), Box<dyn Error>> {
    load_model()
}
