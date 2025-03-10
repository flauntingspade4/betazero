use crate::{MCEdge, MCNode};
// use std::{borrow::BorrowMut, fmt::Debug};

use betazero_nn::{
    board_to_network_input, move_to_probability_index, session_handle::BZSessionHandle,
};
use citron_core::MoveGen;

use ndarray::{Array3, ArrayBase, Axis, Dim, Ix, OwnedRepr};
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
    pub fn new<B: Into<MCNode>>(board: B) -> (Self, NodeIndex) {
        let mut graph = Graph::new();

        let i = graph.add_node(board.into());

        (
            Self {
                graph,
                root_position: i,
            },
            i,
        )
    }

    pub fn moves(&self) -> impl Iterator<Item = EdgeReference<'_, MCEdge>> {
        self.graph.edges(self.root_position)
    }

    pub fn root_position(&self) -> NodeIndex {
        self.root_position
    }

    /// Run a rollout from a given node. Returns the neural network's evaluation
    /// of the most recently expanded leaf node
    pub fn run_rollout_from(
        &mut self,
        node_index: NodeIndex,
        visit_count: usize,
        handle: &BZSessionHandle,
    ) -> f32 {
        // The action with the greatest action value from the current state
        let chosen_edge = {
            let mut outgoing_moves = self.graph.edges(node_index).peekable();

            // This is an unexplored leaf node
            if outgoing_moves.peek().is_none() {
                return self.generate_edges_from(node_index, handle);
            }

            // Choose the action with the greatest action value
            outgoing_moves
                .max_by(|a, b| {
                    a.weight()
                        .action_value(visit_count)
                        .total_cmp(&b.weight().action_value(visit_count))
                })
                .unwrap()
        };
        // println!(
        // "Chosen move {} with p {} and action value {}",
        // chosen_edge.weight().played_move,
        // chosen_edge.weight().prior_probability,
        // chosen_edge.weight().action_value(visit_count)
        // );

        let edge_index = chosen_edge.id();

        let evaluation = self.run_rollout_from(
            chosen_edge.target(),
            chosen_edge.weight().visit_count,
            handle,
        );

        // It's annoying we have to search the graph again to get a mutable
        // reference to the edge, but I can't think of a better way to do it
        self.graph
            .edge_weight_mut(edge_index)
            .unwrap()
            .propagate_value(evaluation);

        evaluation
    }

    /// Generate all the edges from a given node, and the nodes
    /// attached to them
    fn generate_edges_from(
        &mut self,
        node_index: NodeIndex,
        session_hande: &BZSessionHandle,
    ) -> f32 {
        let node = self.graph.node_weight_mut(node_index).unwrap();

        let board = node.board.clone();

        // println!(
        // "Generating moves for side {} on board\n{}",
        // board.to_play(),
        // board
        // );

        let (prior_probabilties, value) = {
            let mut tensor: ArrayBase<OwnedRepr<u64>, Dim<[Ix; 3]>> = Array3::zeros([8, 8, 12]);
            board_to_network_input(&board, &mut tensor);
            let tensor = tensor.insert_axis(Axis(0));
            session_hande.call(Tensor::from(tensor)).unwrap()
        };
        // dbg!((prior_probabilties.shape(), value.shape()));
        // dbg!(&value.shape());
        debug_assert_eq!(value.len(), 3);
        node.value = Some((*value).try_into().unwrap());

        let move_gen = MoveGen::new(&board).into_inner();

        for potential_move in move_gen {
            // println!(
            // "Adding edge for move {} with piece {}",
            // potential_move,
            // potential_move.moved_piece_kind()
            // );
            let new_node = MCNode {
                board: board.make_move(&potential_move).unwrap(),
                value: None,
            };

            let new_node_index = self.graph.add_node(new_node);
            let (x, y, piece_i) = move_to_probability_index(&potential_move);
            self.graph.add_edge(
                node_index,
                new_node_index,
                MCEdge::new(
                    potential_move,
                    prior_probabilties.get(&[0, x as u64, y as u64, piece_i as u64]),
                ),
            );
        }

        0.
    }
}
