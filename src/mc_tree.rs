use std::sync::LazyLock;

use crate::{MCEdge, MCNode};
// use std::{borrow::BorrowMut, fmt::Debug};

use betazero_nn::evaluate_position;
use citron_core::{move_gen::Move, piece::PieceKind, MoveGen};

use petgraph::{
    graph::{EdgeReference, Graph, NodeIndex},
    visit::EdgeRef,
    Directed,
};

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
    pub fn run_rollout_from(&mut self, node_index: NodeIndex, visit_count: usize) -> f64 {
        // The action with the greatest action value from the current state
        let chosen_edge = {
            let mut outgoing_moves = self.graph.edges(node_index).peekable();

            // This is an unexplored leaf node
            if outgoing_moves.peek().is_none() {
                return self.generate_edges_from(node_index);
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
    fn generate_edges_from(&mut self, node_index: NodeIndex) -> f64 {
        let node = self.graph.node_weight_mut(node_index).unwrap();

        let board = node.board.clone();
        let (value, prior_probabilties) = evaluate_position(&board);
        node.value = Some(value);

        let move_gen = MoveGen::new(&board).into_inner();

        for potential_move in move_gen {
            let new_node = MCNode {
                board: board.make_move(&potential_move).unwrap(),
                value: None,
            };

            let new_node_index = self.graph.add_node(new_node);
            let p_index = move_to_probability_index(&potential_move);
            self.graph.add_edge(
                node_index,
                new_node_index,
                MCEdge::new(
                    potential_move,
                    prior_probabilties[p_index.0][p_index.1][p_index.2],
                ),
            );
        }

        value
    }
}

const KNIGHT_DIRECTIONS: [(i16, i16); 8] = [
    (-1, 2),
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
];

static QUEEN_DIRECTIONS: LazyLock<[(i16, i16); 56]> = LazyLock::new(|| {
    let mut queen_moves: [(i16, i16); 56] = [(0, 0); 56];

    let queen_directions: [(i16, i16); 8] = [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (0, -1),
        (-1, 1),
    ];

    for (i, (x_dir, y_dir)) in queen_directions.iter().enumerate() {
        for j in 1..8 {
            queen_moves[i * 7 + j] = (x_dir * j as i16, y_dir * j as i16);
        }
    }

    queen_moves
});

// todo: underpromotion
pub(crate) fn move_to_probability_index(item: &Move) -> (usize, usize, usize) {
    let (from, to) = item.from_to();

    let x_diff = to.x() as i16 - from.x() as i16;
    let y_diff = to.y() as i16 - from.y() as i16;

    let move_index = match item.moved_piece_kind() {
        PieceKind::Knight => match KNIGHT_DIRECTIONS
            .iter()
            .position(|&item| item == (x_diff, y_diff))
        {
            Some(i) => 56 + i,
            None => panic!("Illegal knight move"),
        },
        PieceKind::None => panic!("Illegal none move"),
        _ => match QUEEN_DIRECTIONS
            .iter()
            .position(|&item| item == (x_diff, y_diff))
        {
            Some(i) => i,
            None => panic!("Illegal queen move"),
        },
    };

    (from.x() as usize, from.y() as usize, move_index)
}
