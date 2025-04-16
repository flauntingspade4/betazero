use std::path::Path;

use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs, Status, Tensor};

#[test]
fn flip_call_test() {
    use crate::positions::board_to_network_input;
    use citron_core::Board;
    let handle = BZSessionHandle::load(None);

    let board = &Board::from_fen("k7/2n5/1n6/3r4/4R3/6N1/5N2/7K w - - 0 1").unwrap();
    let board_input = board_to_network_input(&board, citron_core::PlayableTeam::White);
    let output = handle.call(Tensor::from(&board_input)).unwrap();

    let flipped_board = board.make_null_move();
    let flipped_board_input =
        board_to_network_input(&flipped_board, citron_core::PlayableTeam::Black);
    let flipped_output = handle.call(Tensor::from(flipped_board_input)).unwrap();

    assert_eq!(output.0, flipped_output.0);
    assert_eq!(output.1, flipped_output.1);
}

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

        Self { graph, bundle }
    }

    pub fn call(&self, input: Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>), Status> {
        let call_signature = self
            .bundle
            .meta_graph_def()
            .get_signature("call")
            .expect("Signature 'call' not found in saved_mode.pb");

        let input_info = call_signature.get_input("input_1")?;
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
