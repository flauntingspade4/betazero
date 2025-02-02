use citron_core::{move_gen::Move, Board, PlayableTeam};

use crate::mc_tree::move_to_probability_index;

pub struct GameRecord {
    boards: Vec<Board>,
    moves: Vec<[[[usize; 64]; 8]; 8]>,
}

impl GameRecord {
    pub fn new() -> Self {
        Self {
            boards: Vec::new(),
            moves: Vec::new(),
        }
    }

    pub fn add_move(&mut self, board: &Board, moves: &[(usize, Move)]) {
        self.boards.push(board.clone());

        let mut probability_matrix = [[[0; 64]; 8]; 8];

        for (move_probability, m) in moves.iter() {
            let (x, y, i) = move_to_probability_index(m);

            probability_matrix[x][y][i] = *move_probability;
        }

        self.moves.push(probability_matrix);
    }

    pub fn finish(
        self,
        winning_team: PlayableTeam,
    ) -> (Vec<Board>, Vec<[[[usize; 64]; 8]; 8]>, Vec<i16>) {
        let winning = if winning_team == PlayableTeam::White {
            [1i16, -1]
        } else {
            [-1, 1]
        }
        .into_iter()
        .cycle()
        .take(self.boards.len());

        (self.boards, self.moves, winning.collect())
    }
}
