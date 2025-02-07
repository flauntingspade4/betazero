use citron_core::{move_gen::Move, Board, Team};

use crate::move_to_probability_index;

pub struct GameRecord {
    boards: Vec<Board>,
    moves: Vec<[[[f32; 64]; 8]; 8]>,
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

        let max = moves
            .iter()
            .max_by(|(m_0, _), (m_1, _)| m_0.cmp(m_1))
            .expect("No possible moves")
            .0;

        let moves = moves
            .iter()
            .map(|(visit_count, m)| (*visit_count as f32 / max as f32, m));

        let mut probability_matrix = [[[0.; 64]; 8]; 8];

        for (move_probability, m) in moves {
            let (x, y, i) = move_to_probability_index(m);

            probability_matrix[x][y][i] = move_probability;
        }

        self.moves.push(probability_matrix);
    }

    pub fn finish(self, winning_team: Team) -> (Vec<Board>, Vec<[[[f32; 64]; 8]; 8]>, Vec<i16>) {
        let won = match winning_team {
            Team::White => [1, -1],
            Team::Black => [-1, 1],
            Team::Neither => [0, 0],
        }
        .into_iter()
        .cycle()
        .take(self.boards.len());

        (self.boards, self.moves, won.collect())
    }
}

pub struct BoardRecord {
    pub board: Board,
    pub moves: [[[f32; 64]; 8]; 8],
    pub won: i16,
}
