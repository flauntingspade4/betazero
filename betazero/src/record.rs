use ndarray::{Array4, Axis};
use serde::{Deserialize, Serialize};

use citron_core::{move_gen::Move, nn_util::board_to_network_input, Board, Team};
use serde_with::serde_as;

use crate::positions::move_to_probability_index;

/// A record of a game, saving the visit
/// count for training
pub struct GameRecord {
    boards: Vec<Array4<u64>>,
    moves: Vec<[[[f32; 64]; 8]; 8]>,
}

impl GameRecord {
    pub fn new() -> Self {
        Self {
            boards: Vec::new(),
            moves: Vec::new(),
        }
    }

    /// Adds a move to the record. `moves` is a list
    /// of moves and the number of times they were visited
    /// during rollouts
    pub fn add_move(&mut self, board: &Board, moves: &[(Move, usize, f32, f32)]) {
        let input = board_to_network_input(board, board.to_play());
        let input = input.insert_axis(Axis(0));
        self.boards.push(input);

        let total: usize = moves
            .iter()
            .map(|(_, visit_count, _, _)| *visit_count)
            .sum();

        let moves = moves
            .iter()
            .map(|(m, visit_count, _, _)| (*visit_count as f32 / total as f32, m));

        let mut probability_matrix = [[[0.; 64]; 8]; 8];

        for (move_probability, m) in moves {
            let (x, y, i) = move_to_probability_index(m, board.to_play());

            probability_matrix[x][y][i] = move_probability;
        }

        self.moves.push(probability_matrix);
    }

    /// Finishes the game and returns a list of training
    /// examples
    pub fn finish(self, winning_team: Team) -> Vec<BoardRecord> {
        let won = match winning_team {
            Team::White => [[1.0f32, 0., 0.], [0., 0., 1.]],
            Team::Black => [[0., 0., 1.], [1., 0., 0.]],
            Team::Neither => [[0., 1., 0.], [0., 1., 0.]],
        }
        .into_iter()
        .cycle();

        println!(
            "{} won",
            match winning_team {
                Team::White => "White",
                Team::Black => "Black",
                Team::Neither => "Neither",
            }
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
#[derive(Deserialize, Serialize, Debug)]
pub struct BoardRecord {
    // #[serde_as(as = "[[[_; 12]; 8]; 8]")]
    pub board: Array4<u64>,
    #[serde_as(as = "[[[_; 64]; 8]; 8]")]
    pub moves: [[[f32; 64]; 8]; 8],
    pub won: [f32; 3],
    pub move_number: u64,
}
