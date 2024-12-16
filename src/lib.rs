mod mc_tree;

use citron_core::move_gen::{Move, MoveGen};
use citron_core::Board;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

const EXPLORATION_PARAMETER: f64 = std::f64::consts::SQRT_2;

#[derive(Debug, Clone, Eq)]
pub struct MCNode<O>
where
    O: Clone + Debug,
{
    board: Board,
    neural_net_output: Option<O>,
}

impl<O> PartialEq<Self> for MCNode<O>
where
    O: Clone + Debug,
{
    fn eq(&self, other: &Self) -> bool {
        self.board == other.board
    }
}

impl<O> From<Board> for MCNode<O>
where
    O: Clone + Debug,
{
    fn from(value: Board) -> Self {
        Self {
            board: value,
            neural_net_output: None,
        }
    }
}

impl<O> Hash for MCNode<O>
where
    O: Clone + Debug,
{
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
                * ((parent_visit_count as f64).ln() / self.visit_count as f64).sqrt()
    }
}
