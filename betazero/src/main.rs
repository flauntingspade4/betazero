#![warn(clippy::pedantic, clippy::all, clippy::nursery)]

use std::{
    fs::File,
    hash::{Hash, Hasher},
    io::BufWriter,
    path::PathBuf,
    time::Instant,
};

use chrono::Local;
use rand::seq::IndexedRandom;

use serde_pickle::SerOptions;

use citron_core::{move_gen::Move, piece::PieceKind, Board, PlayableTeam, Team};

mod mc_tree;
pub mod positions;
mod record;
mod session_handle;

use mc_tree::MCTree;
use record::{BoardRecord, GameRecord};
use session_handle::BZSessionHandle;

const EXPLORATION_PARAMETER: f32 = std::f32::consts::SQRT_2;

/// Self plays a game from the starting position, returning a list
/// of records for each move
#[must_use]
pub fn self_play_game(rollout_limit: usize, handle: &BZSessionHandle) -> Vec<BoardRecord> {
    let starting_board = Board::new();

    self_play_game_from(starting_board, rollout_limit, handle)
}

/// Self plays from a given position, returning a list of records
/// for each move
pub fn self_play_game_from(
    mut board: Board,
    rollout_limit: usize,
    handle: &BZSessionHandle,
) -> Vec<BoardRecord> {
    let mut rng = rand::rng();
    let mut game_record = GameRecord::new();

    loop {
        match board.result() {
            Some(PlayableTeam::White) | Some(PlayableTeam::Black) => break,
            None => {
                let tree = analyse_position(board.clone(), rollout_limit, handle);

                let potential_moves: Vec<(Move, usize, f32, f32)> = tree
                    .moves()
                    .map(|m| {
                        (
                            m.weight().played_move(),
                            m.weight().visit_count,
                            m.weight().average_value,
                            m.weight().prior_probability,
                        )
                    })
                    .collect();

                let (chosen_move, visits, v, p) = if let Some((winning_move, visits, v, p)) =
                    potential_moves
                        .iter()
                        .find(|(m, _, _, _)| m.captured_piece_kind() == PieceKind::King)
                {
                    println!(
                        "Playing winning king move with probability {} ({} visits)",
                        v, visits
                    );
                    (winning_move, visits, v, p)
                } else {
                    match potential_moves
                        .choose_weighted(&mut rng, |(_, visit_count, _, _)| *visit_count)
                    {
                        Ok((m, visits, v, p)) => (m, visits, v, p),
                        // This should never happen, but if there is never a possible move
                        // then it should be treated as stalemate (Maybe)
                        Err(_) => break,
                    }
                };

                game_record.add_move(&board, &potential_moves);
                board = board.make_move(&chosen_move);
                println!(
                    "Next move played with v={}, p={} and {} visits\n{}",
                    v, p, visits, board
                );
            }
        }
    }

    game_record.finish(board.result().map_or(Team::Neither, Into::into))
}

#[must_use]
pub fn analyse_position(board: Board, rollout_limit: usize, handle: &BZSessionHandle) -> MCTree {
    let mut tree = MCTree::new(board);

    let start = Instant::now();
    for i in 0..rollout_limit {
        tree.run_rollout_from_root(i + 1, handle).unwrap();
        // println!("Done rollout {i}");
    }
    println!("Move took {}ms", (Instant::now() - start).as_millis());

    tree
}

#[test]
fn self_play_test() {
    use serde_pickle::SerOptions;
    use std::{fs::File, io::BufWriter};

    let handle = BZSessionHandle::load(None);

    println!("Ready for self play");

    let mut games = Vec::new();

    for i in 0..2 {
        println!("On game {i}");
        games.extend(self_play_game(250, &handle));
    }

    let f = File::create("games2.json").unwrap();
    let mut f = BufWriter::new(f);
    serde_pickle::to_writer(&mut f, &games, SerOptions::new()).unwrap();
}

fn main() {
    let handle = BZSessionHandle::load(None);
    let mut games = Vec::new();

    /*let mut receivers = Vec::new();
    let mut threads = Vec::new();
    // println!("Ready for self play");
    for thread in 0..2 {
        let (w, r) = std::sync::mpsc::channel();
        receivers.push(r);
        let t = std::thread::spawn(move || {
            let handle = BZSessionHandle::load(None);
            for i in 0..7 {
                w.send(self_play_game(1000, &handle)).unwrap();
                println!("Sent game {} from thread {}", i, thread);
            }
        });

        threads.push(t);
    }

    for t in threads {
        t.join().unwrap();
    }

    for r in receivers {
        while let Ok(game) = r.recv() {
            println!("Received game of len {}", game.len());
            games.extend(game);
        }
    }*/
    for i in 0..10 {
        println!("On game {i}");
        games.extend(self_play_game(1500, &handle));
    }

    println!("Writing {} many moves", games.len());
    let local = Local::now();
    let path = PathBuf::from(&format!("games/{}game.pickle", local.format("%Hh%Mm")));

    let f = File::create(path).unwrap();
    let mut f = BufWriter::new(f);
    serde_pickle::to_writer(&mut f, &games, SerOptions::new()).unwrap();

    let f = File::create("latest.pickle").unwrap();
    let mut f = BufWriter::new(f);
    serde_pickle::to_writer(&mut f, &games, SerOptions::new()).unwrap();
}

#[test]
fn yet_another_test() {
    let input =
        ndarray::Array::from_shape_fn((1, 8, 8, 12), |(_, _, _, z)| if z == 0 { 1 } else { 0 });
    println!("{:?}", input);
    // let input = Array3::from([[[1u64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; 8]; 8]);
    let handle = BZSessionHandle::load(None);

    // println!("{:?}", handle.call(tensorflow::Tensor::from(input)));

    analyse_position(Board::new(), 3, &handle);
}

#[test]
fn move_gen_test() {
    use citron_core::{piece::PieceKind, MoveGen};
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

#[derive(Debug, Clone)]
pub struct MCNode {
    board: Board,
    /// A neural network evaluation of the current position.
    /// When set to `None`, moves from this position have not
    /// yet been generated
    outputs: Option<[f32; 3]>,
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
            outputs: None,
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
    #[must_use]
    pub const fn new(played_move: Move, prior_probability: f32) -> Self {
        Self {
            played_move,
            visit_count: 0,
            average_value: 0.,
            total_value: 0.,
            prior_probability,
        }
    }

    #[must_use]
    pub fn played_move(&self) -> Move {
        self.played_move.clone()
    }

    pub fn propagate_value(&mut self, value: f32) {
        self.visit_count += 1;
        self.total_value += value;
        self.average_value = self.total_value / self.visit_count as f32;
    }

    pub fn action_value(&self, parent_visit_count: usize) -> f32 {
        (EXPLORATION_PARAMETER * self.prior_probability).mul_add(
            (parent_visit_count as f32).sqrt() / (1. + self.visit_count as f32),
            self.average_value,
        )
    }
}
