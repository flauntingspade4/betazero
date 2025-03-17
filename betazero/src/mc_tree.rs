use crate::{
    positions::{board_to_network_input, move_to_probability_index},
    BZSessionHandle, MCEdge, MCNode,
};

use citron_core::{MoveGen, PlayableTeam};

use petgraph::{
    graph::{EdgeReference, Graph, NodeIndex},
    visit::EdgeRef,
    Directed,
};
use tensorflow::Tensor;

pub struct MCTree {
    graph: Graph<MCNode, MCEdge, Directed>,
    root_position: NodeIndex,
}

impl MCTree {
    /// Creates a new monte-carlo tree with the given board
    /// as the noot rode
    pub fn new<B: Into<MCNode>>(board: B) -> Self {
        let mut graph = Graph::new();

        let i = graph.add_node(board.into());

        Self {
            graph,
            root_position: i,
        }
    }

    /// Returns an iterate over all of the moves from the root
    /// position. This should be called after many runs of
    /// [`Self::run_rollout_from_root`], and a move selected
    /// according to the number of times each move has been visited
    pub fn moves(&self) -> impl Iterator<Item = EdgeReference<'_, MCEdge>> {
        self.graph.edges(self.root_position)
    }

    pub const fn root_position(&self) -> NodeIndex {
        self.root_position
    }

    /// Runs a pass of [`Self::run_rollout_from`] from the root node.
    /// Visit count should equal the number of times this method has been
    /// called for this tree.
    pub fn run_rollout_from_root(
        &mut self,
        visit_count: usize,
        handle: &BZSessionHandle,
    ) -> Result<f32, ()> {
        self.run_rollout_from(self.root_position, visit_count, handle)
    }

    /// Run a rollout from a given node. This will traverse the
    /// tree according to the neural network's evaluation until
    /// it reaches an unexplored position (a new leaf node). At
    /// This point it will call [`Self::generate_edges_from`] on this
    /// new leaf node, and return the evaluation from that
    pub fn run_rollout_from(
        &mut self,
        node_index: NodeIndex,
        visit_count: usize,
        handle: &BZSessionHandle,
    ) -> Result<f32, ()> {
        // The action with the greatest action value from the current state

        let mut outgoing_moves = self.graph.edges(node_index).peekable();

        // This is an unexplored leaf node
        if outgoing_moves.peek().is_none() {
            return self.generate_edges_from(node_index, handle);
        }

        // Choose the action with the greatest action value
        let chosen_edge = outgoing_moves
            .max_by(|a, b| {
                a.weight()
                    .action_value(visit_count)
                    .total_cmp(&b.weight().action_value(visit_count))
            })
            .unwrap();
        let edge_index = chosen_edge.id();

        let board = &self.graph.node_weight(node_index).unwrap().board;
        let to_play = board.to_play();
        let evaluation = match board.result() {
            Some(t @ PlayableTeam::White) | Some(t @ PlayableTeam::Black) => {
                match t.compare(&to_play.into()) {
                    citron_core::TeamComparison::Same => Ok(1.),
                    citron_core::TeamComparison::Different => Ok(-1.),
                    citron_core::TeamComparison::None => unreachable!(),
                }
            }
            None => Ok(-self.run_rollout_from(
                chosen_edge.target(),
                chosen_edge.weight().visit_count,
                handle,
            )?),
        };

        if let Ok(ev) = evaluation {
            // It's annoying we have to search the graph again to get a mutable
            // reference to the edge, but I can't think of a better way to do it
            self.graph
                .edge_weight_mut(edge_index)
                .unwrap()
                .propagate_value(ev);
        }

        evaluation
    }

    /// This should be called on a position that has been inserted into
    /// the tree, but not yet evaluated by the neural network, i.e. a leaf
    /// node found by a run of [`Self::run_rollout_from`]. This method will then
    /// calculate all the possible moves from the position and insert them
    /// with their resulting board states into the tree, before evaluating
    /// the current position with the given [`BZSessionHandle`] and updating
    /// the tree accordingly
    fn generate_edges_from(
        &mut self,
        node_index: NodeIndex,
        session_hande: &BZSessionHandle,
    ) -> Result<f32, ()> {
        let node = self.graph.node_weight_mut(node_index).unwrap();

        let board = node.board.clone();
        let move_gen = MoveGen::new(&board).into_inner();

        let (c_prior_probabilties, c_values) = session_hande
            .call(Tensor::from(board_to_network_input(
                &board,
                board.to_play(),
            )))
            .unwrap();
        let value = (*c_values).try_into().unwrap();
        node.outputs = Some(value);

        for potential_move in move_gen {
            let new_board = board.make_move(&potential_move);
            let new_node = MCNode {
                board: new_board,
                outputs: None,
            };

            let new_node_index = self.graph.add_node(new_node);
            let (x, y, piece_i) = move_to_probability_index(&potential_move, board.to_play());
            self.graph.add_edge(
                node_index,
                new_node_index,
                MCEdge::new(
                    potential_move,
                    c_prior_probabilties.get(&[0, x as u64, y as u64, piece_i as u64]),
                ),
            );
        }

        match board.result() {
            Some(t @ PlayableTeam::White) | Some(t @ PlayableTeam::Black) => {
                match t.compare(&board.to_play().into()) {
                    citron_core::TeamComparison::Same => Ok(1.),
                    citron_core::TeamComparison::Different => Ok(-1.),
                    citron_core::TeamComparison::None => unreachable!(),
                }
            }
            None => Ok(match board.to_play() {
                PlayableTeam::White => value[0] - value[2],
                PlayableTeam::Black => value[2] - value[0],
            }),
        }
    }
}

#[test]
fn speed_comparison() {
    let handle = BZSessionHandle::load(None);

    let board = Board::new();
    let move_gen = MoveGen::new(&board).into_inner();
    let input = board_to_network_input(&board, PlayableTeam::White);

    let mut outputs = Vec::new();
    let start = Instant::now();
    for _ in 0..20 {
        outputs.push(handle.call(input.clone().into()).unwrap());
    }
    println!("Took {}ms", (Instant::now() - start).as_millis());

    let board = Board::new();
    let move_gen = MoveGen::new(&board).into_inner();

    let mut input: ndarray::ArrayBase<ndarray::OwnedRepr<u64>, ndarray::Dim<[usize; 4]>> =
        Array4::zeros((2, 8, 8, 12));

    let moves = move_gen
        .into_iter()
        .take(2)
        .map(|m| (board.make_move(&m), m))
        .collect::<Vec<_>>();

    let start = Instant::now();

    for ((board, _), axis) in moves.iter().zip(input.axis_iter_mut(Axis(0))) {
        board_to_network_input(board, board.to_play())
            .to_shape((8, 8, 12))
            .unwrap()
            .assign_to(axis);
    }

    let (p_values, v_values) = handle.call(input.into()).unwrap();
    handle.call(Tensor::from(board_to_network_input(
        &board,
        board.to_play(),
    )));
    println!("Took {}ms", (Instant::now() - start).as_millis());

    println!("Final output\n{:?}", p_values);
    for values in p_values.chunks(8 * 8 * 64) {
        println!("Value {:?}", values);
    }
}
