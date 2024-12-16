use crate::{MCEdge, MCNode};
use std::{borrow::BorrowMut, fmt::Debug};

use betazero_nn::evaluate_position;
use citron_core::MoveGen;

use petgraph::{
    data::DataMapMut,
    graph::{Graph, NodeIndex},
    visit::EdgeRef,
    Directed,
};

pub struct MCTree<O>
where
    O: Clone + Debug,
{
    graph: Graph<MCNode<O>, MCEdge, Directed>,
}

impl<O> MCTree<O>
where
    O: Clone + Debug,
{
    pub fn new<B: Into<MCNode<O>>>(board: B) -> Self {
        let mut graph = Graph::new();

        graph.add_node(board.into());

        Self { graph }
    }

    /// Run a rollout from a given node. Returns the neural network's evaluation
    /// of the most recently expanded leaf node
    pub fn run_rollout_from(&mut self, node: NodeIndex, visit_count: usize) -> f64 {
        // The action with the greatest action value from the current state
        let chosen_edge = {
            let mut outgoing_moves = self.graph.edges(node).peekable();

            // This is an unexplored leaf node
            if outgoing_moves.peek().is_none() {
                return self.generate_edges_from(node);
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
        let edge_index = chosen_edge.id();

        let evaluation =
            self.run_rollout_from(chosen_edge.target(), chosen_edge.weight().visit_count);

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
    fn generate_edges_from(&mut self, node: NodeIndex) -> f64 {
        let board = self.graph.node_weight(node).unwrap().board.clone();

        let move_gen = MoveGen::new(&board);

        let (value, prior_probabilties) = evaluate_position(&board);

        for potential_move in move_gen.into_inner() {
            let new_node = MCNode {
                board: board.make_move(&potential_move).unwrap(),
                neural_net_output: None,
            };

            let new_node_index = self.graph.add_node(new_node);
            self.graph.add_edge(
                node,
                new_node_index,
                MCEdge::new(potential_move, prior_probabilties[0][0][0]),
            );
        }

        value
    }
}
