mod mc_tree;
mod record;

use citron_core::move_gen::Move;
use citron_core::Board;
use mc_tree::MCTree;
use rand::seq::IndexedRandom;
use record::GameRecord;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

const EXPLORATION_PARAMETER: f64 = std::f64::consts::SQRT_2;

pub fn self_play_from(mut board: Board, rollout_limit: usize) {
    let mut rng = rand::rng();
    let mut game_record = GameRecord::new();

    while board.result().is_none() {
        let tree = analyse_position(board.clone(), rollout_limit);

        let potential_moves: Vec<(usize, Move)> = tree
            .moves()
            .map(|m| (m.weight().visit_count, m.weight().played_move.clone()))
            .collect();

        let chosen_move = potential_moves
            .choose_weighted(&mut rng, |(visit_count, _)| *visit_count)
            .expect("Trying to choose a move from an empty list")
            .clone();

        game_record.add_move(&board, &potential_moves);
        board = board.make_move(&chosen_move.1).unwrap();
    }

    let _ = game_record.finish(board.result().unwrap());
}

pub fn analyse_position(board: Board, rollout_limit: usize) -> MCTree {
    let (mut tree, node) = MCTree::new(board);

    for i in 0..rollout_limit {
        tree.run_rollout_from(node, i);
    }

    tree
}

#[derive(Debug, Clone)]
pub struct MCNode {
    board: Board,
    /// A neural network evaluation of the current position.
    /// When set to `None`, moves from this position have not
    /// yet been generated
    value: Option<f64>,
}

impl PartialEq<Self> for MCNode {
    fn eq(&self, other: &Self) -> bool {
        self.board == other.board
    }
}

impl From<Board> for MCNode {
    fn from(value: Board) -> Self {
        Self {
            board: value,
            value: None,
        }
    }
}

impl Hash for MCNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.board.hash());
    }
}

#[derive(Debug, Clone)]
pub struct MCEdge {
    played_move: Move,
    visit_count: usize,
    average_value: f64,
    total_value: f64,
    prior_probability: f64,
}

impl MCEdge {
    pub fn new(played_move: Move, prior_probability: f64) -> Self {
        Self {
            played_move,
            visit_count: 0,
            average_value: 0.,
            total_value: 0.,
            prior_probability,
        }
    }

    pub fn played_move(&self) -> Move {
        self.played_move.clone()
    }

    pub fn propagate_value(&mut self, value: f64) {
        self.visit_count += 1;
        self.total_value += value;
        self.average_value = self.total_value / self.visit_count as f64;
    }

    pub fn action_value(&self, parent_visit_count: usize) -> f64 {
        self.average_value
            + EXPLORATION_PARAMETER
                * self.prior_probability
                * ((parent_visit_count as f64).sqrt() / (1. + self.visit_count as f64))
    }
}
