use std::{
    fs::{read_to_string, File},
    io::BufWriter,
};

use ndarray::Array3;

use citron_core::{
    nn::{add_move_information_to_nn_input, board_to_network_input},
    Board, PlayableTeam,
};
use pgnparse::parser::parse_pgn_to_rust_struct;
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use serde_pickle::SerOptions;

const POSITIONS_PER_FILE: usize = 100_000;

pub fn main() {
    let input = read_to_string(r"games\CCRL.40-2.Archive.[2165313].pgn").unwrap();
    let mut input = input.as_str();
    let mut rng = rand::rng();

    let mut white_won = FileRecord::new("white_won".to_string(), "white_won".to_string());
    let mut black_won = FileRecord::new("black_won".to_string(), "black_won".to_string());

    for i in 0.. {
        if i % 1000 == 0 {
            println!("On game {i}");
        }

        let mut output = if let Some(next_game_index) = input.find("\n[Event") {
            let output = parse_pgn_to_rust_struct(&input[..next_game_index]);
            input = &input[1 + next_game_index..];
            output
        } else {
            break;
        };

        let won_team = match output.get_header("Result").as_str() {
            "1-0" => PlayableTeam::White,
            "0-1" => PlayableTeam::Black,
            "1/2-1/2" => continue,
            a => {
                println!("Incorrect victory team {a}");
                break;
            }
        };

        for played_move in output.moves.iter().skip(7).choose_multiple(&mut rng, 6) {
            let board = Board::from_fen(&played_move.fen_before).unwrap();

            let mut r = board_to_network_input(&board, PlayableTeam::White);
            add_move_information_to_nn_input(&board, &mut r);
            // ඞ ඞ ඞ

            match won_team {
                PlayableTeam::White => white_won.add_position(r),
                PlayableTeam::Black => black_won.add_position(r),
            }
        }
    }

    white_won.save();
    black_won.save();
}

#[derive(Serialize, Deserialize)]
pub struct AERecord {
    board: Array3<f32>,
}

pub struct FileRecord {
    positions: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>>,
    counter: usize,
    name: String,
    dir: String,
}

impl FileRecord {
    pub fn new(name: String, dir: String) -> Self {
        Self {
            positions: Vec::with_capacity(POSITIONS_PER_FILE),
            counter: 0,
            name,
            dir,
        }
    }

    pub fn add_position(
        &mut self,
        position: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 3]>>,
    ) {
        self.positions.push(position);
        if self.positions.len() >= POSITIONS_PER_FILE {
            self.save();
        }
    }

    pub fn save(&mut self) {
        let file = File::create(format!(
            "{}/{}-{}.pickle",
            self.dir, self.name, self.counter
        ))
        .unwrap();

        let mut f = BufWriter::new(file);
        serde_pickle::to_writer(&mut f, &self.positions, SerOptions::new()).unwrap();

        self.positions.clear();
        self.counter += 1;
    }
}
