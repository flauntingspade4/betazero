use ndarray::Array3;
use serde::{Deserialize, Serialize};

use citron_core::{move_gen::Move, pgn::Pgn, Board, Team};
use serde_with::serde_as;

use crate::{board_to_network_input, move_to_probability_index};

/// A record of a game, saving the visit
/// count for training
pub struct GameRecord {
    boards: Vec<Array3<u64>>,
    moves: Vec<[[[f32; 64]; 8]; 8]>,
    pgn: Pgn,
}

impl GameRecord {
    pub fn new() -> Self {
        Self {
            boards: Vec::new(),
            moves: Vec::new(),
            pgn: Pgn::new(),
        }
    }

    /// Adds a move to the record. `moves` is a list
    /// of moves and the number of times they were visited
    /// during rollouts
    pub fn add_move(&mut self, board: &Board, moves: &[(Move, usize)], played_move: &Move) {
        self.boards.push(board_to_network_input(board));
        self.pgn.add_move(&played_move);

        let total: usize = moves.iter().map(|(_, visit_count)| *visit_count).sum();

        let moves = moves
            .iter()
            .map(|(m, visit_count)| (*visit_count as f32 / total as f32, m, *visit_count));

        let mut probability_matrix = [[[0.; 64]; 8]; 8];

        println!(
            "Following moves for this board\n{}\nBoard is also {}\nMax visit count is {}",
            board,
            self.boards.last().unwrap(),
            total
        );

        for (move_probability, m, visit_count) in moves {
            let (x, y, i) = move_to_probability_index(m);
            println!(
                "Move {} has visit count {} and visit probability {}",
                m, visit_count, move_probability
            );

            probability_matrix[x][y][i] = move_probability;
        }

        self.moves.push(probability_matrix);
    }

    /// Finishes the game and returns a list of training
    /// examples
    pub fn finish(self, winning_team: Team) -> Vec<BoardRecord> {
        let won = std::iter::repeat(match winning_team {
            Team::White => [1., 0., 0.],
            Team::Black => [0., 0., 1.],
            Team::Neither => [0., 1., 0.],
        });
        println!(
            "Game finished with the following pgn:\n{}",
            self.pgn.finish()
        );
        self.boards
            .into_iter()
            .zip(self.moves.into_iter())
            .zip(won)
            .enumerate()
            .map(|(i, ((board, moves), won))| BoardRecord {
                board,
                moves,
                won,
                move_number: i as u64,
            })
            .collect()
    }
}

#[serde_as]
#[derive(Deserialize, Serialize)]
pub struct BoardRecord {
    // #[serde_as(as = "[[[_; 12]; 8]; 8]")]
    pub board: Array3<u64>,
    #[serde_as(as = "[[[_; 64]; 8]; 8]")]
    pub moves: [[[f32; 64]; 8]; 8],
    pub won: [f32; 3],
    pub move_number: u64,
}
