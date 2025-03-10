// extern crate tensorflow;

use betazero_nn::{
    record::{BoardRecord, GameRecord},
    session_handle::BZSessionHandle,
};
use citron_core::{move_gen::Move, piece::PieceKind, Board, MoveGen, Team};
use rand::seq::IndexedRandom;
use std::{
    hash::{Hash, Hasher},
    path::Path,
};

mod mc_tree;

use mc_tree::MCTree;

const EXPLORATION_PARAMETER: f32 = std::f32::consts::SQRT_2;

pub fn self_play_from(
    mut board: Board,
    rollout_limit: usize,
    handle: &BZSessionHandle,
) -> Vec<BoardRecord> {
    let mut rng = rand::rng();
    let mut game_record = GameRecord::new();

    while board.result().is_none() {
        let tree = analyse_position(board.clone(), rollout_limit, &handle);

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

    game_record.finish(match board.result() {
        Some(t) => t.into(),
        None => Team::Neither,
    })
}

#[test]
fn self_play_test() {
    let board = Board::new();
    let handle = BZSessionHandle::load(None);

    println!("Ready for self play");

    self_play_from(board, 5, &handle);
}

#[test]
fn model_2_test() {
    let board = Board::new();
    let handle = BZSessionHandle::load(None);

    let (mut tree, node) = MCTree::new(board);

    for i in 0..10 {
        println!("Rollout {i}");
        tree.run_rollout_from(node, i, &handle);
    }
}

#[test]
fn move_gen_test() {
    /*println!("{:b}", 9259542123273814144u64);

    let board =
        Board::from_fen("rnbqkbnr/pppppp1p/8/6p1/P7/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    println!("{board}");
    println!("{:b}", board.blockers());

    let moves = MoveGen::new(&board);

    for p_move in moves.into_inner().iter() {
        println!(
            "{p_move} with piece {} capturing {}",
            p_move.moved_piece_kind(),
            p_move.captured_piece_kind()
        );
    }*/

    let board =
        Board::from_fen("rnbqkbnr/pppppp1p/8/6p1/P7/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();

    let move_gen = MoveGen::new(&board).into_inner();
    // let blockers = board.get_occupied();
    // println!("{:b}", blockers);
    // let bishops = board.pieces[0][PieceKind::Bishop as usize];
    // println!("{:b}", bishops);

    for bishop_move in move_gen
        .iter()
        .filter(|p| p.moved_piece_kind() == PieceKind::Bishop)
    {
        println!("Bishop move {}", bishop_move);
    }

    println!("{}", board);
}

pub fn analyse_position(board: Board, rollout_limit: usize, handle: &BZSessionHandle) -> MCTree {
    let (mut tree, node) = MCTree::new(board);

    for i in 0..rollout_limit {
        tree.run_rollout_from(node, i, handle);
    }

    tree
}

#[derive(Debug, Clone)]
pub struct MCNode {
    board: Board,
    /// A neural network evaluation of the current position.
    /// When set to `None`, moves from this position have not
    /// yet been generated
    value: Option<[f32; 3]>,
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
    average_value: f32,
    total_value: f32,
    prior_probability: f32,
}

impl MCEdge {
    pub fn new(played_move: Move, prior_probability: f32) -> Self {
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

    pub fn propagate_value(&mut self, value: f32) {
        self.visit_count += 1;
        self.total_value += value;
        self.average_value = self.total_value / self.visit_count as f32;
    }

    pub fn action_value(&self, parent_visit_count: usize) -> f32 {
        self.average_value
            + EXPLORATION_PARAMETER
                * self.prior_probability
                * ((parent_visit_count as f32).sqrt() / (1. + self.visit_count as f32))
    }
}
