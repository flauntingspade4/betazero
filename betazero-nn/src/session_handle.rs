use std::path::Path;

use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Shape, Status, Tensor};

pub struct BZSessionHandle {
    graph: Graph,
    bundle: SavedModelBundle,
}

impl BZSessionHandle {
    pub fn load(path: Option<&Path>) -> Self {
        let path: &Path = path.unwrap_or(&Path::new("model"));

        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, path)
            .expect("Can't load saved model");

        let sign_def = bundle.meta_graph_def().signatures();
        println!("SIGNATURES: {:#?}", sign_def);

        // dbg!(bundle.meta_graph_def());
        // let init_signature = bundle
        // .meta_graph_def()
        // .get_signature("__saved_model_init_op")
        // .unwrap();
        // println!("Init signature name = {}", init_signature.method_name());
        // let op_init = graph
        // .operation_by_name_required(init_signature.method_name())
        // .unwrap();
        // let mut init_step = SessionRunArgs::new();
        // init_step.add_target(&op_init);
        // bundle.session.run(&mut init_step).unwrap();

        // println!("Performed init op");

        Self { graph, bundle }
    }

    pub fn call(&self, input: Tensor<u64>) -> Result<(Tensor<f32>, Tensor<f32>), Status> {
        debug_assert_eq!(
            input.shape(),
            Shape::new(Some(vec![Some(1), Some(8), Some(8), Some(12)]))
        );
        let call_signature = self
            .bundle
            .meta_graph_def()
            .get_signature("call")
            .expect("Signature 'call' not found in saved_mode.pb");

        let input_info = call_signature.get_input("board")?;
        let input_op = self
            .graph
            .operation_by_name(&input_info.name().name)?
            .unwrap();

        let policy_info = call_signature.get_output("policy_output")?;
        let policy_op = self
            .graph
            .operation_by_name(&policy_info.name().name)?
            .unwrap();

        let value_info = call_signature.get_output("value_output")?;
        let value_op = self
            .graph
            .operation_by_name(&value_info.name().name)?
            .unwrap();

        let mut call_step = SessionRunArgs::new();
        call_step.add_feed(&input_op, 0, &input);
        let (policy, value) = (
            call_step.request_fetch(&policy_op, 0),
            call_step.request_fetch(&value_op, 1),
        );
        self.bundle.session.run(&mut call_step)?;

        Ok((call_step.fetch(policy)?, call_step.fetch(value)?))
    }
}
