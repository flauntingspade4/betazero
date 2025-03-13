use citron_core::{move_gen::Move, piece::PieceKind, Board, PlayableTeam, Position};
use ndarray::Array3;
use std::sync::LazyLock;

pub mod record;
pub mod session_handle;

pub use session_handle::BZSessionHandle;

/// Calculates an array from the board that can
/// then be used with [`BZSessionHandle`]
pub fn board_to_network_input(board: &Board) -> Array3<u64> {
    let mut array = Array3::zeros([8, 8, 12]);

    let mut piece_map = board.pieces();
    // Iterate over each team and piece kind
    for &team in PlayableTeam::teams().iter() {
        for &piece_type in PieceKind::kinds().iter() {
            // Add each piece for that team and kind to the array
            while piece_map[team as usize][piece_type as usize] != 0 {
                let bitmap =
                    citron_core::magic::pop_lsb(&mut piece_map[team as usize][piece_type as usize]);
                let position = Position::from_bitmap(1 << bitmap);

                array[[
                    position.x() as usize,
                    position.y() as usize,
                    team as usize * 6 + piece_type as usize,
                ]] = 1;
            }
        }
    }

    array
}

const KNIGHT_MOVES: [(i16, i16); 8] = [
    (-1, 2),
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
];

static SLIDING_MOVES: LazyLock<[(i16, i16); 56]> = LazyLock::new(|| {
    let mut queen_moves: [(i16, i16); 56] = [(0, 0); 56];

    let queen_directions: [(i16, i16); 8] = [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ];

    for (i, (x_dir, y_dir)) in queen_directions.iter().enumerate() {
        for j in 1..8 {
            queen_moves[i * 7 + j - 1] = (x_dir * j as i16, y_dir * j as i16);
        }
    }

    queen_moves
});

// todo: underpromotion
pub fn move_to_probability_index(item: &Move) -> (usize, usize, usize) {
    let (from, to) = item.from_to();

    let x_diff = to.x() as i16 - from.x() as i16;
    let y_diff = to.y() as i16 - from.y() as i16;

    let move_index = match item.moved_piece_kind() {
        PieceKind::None => panic!("Illegal none move"),
        PieceKind::Knight => match KNIGHT_MOVES
            .iter()
            .position(|&item| item == (x_diff, y_diff))
        {
            Some(i) => 56 + i,
            None => panic!("Illegal knight move"),
        },
        _ => match SLIDING_MOVES
            .iter()
            .position(|&item| item == (x_diff, y_diff))
        {
            Some(i) => i,
            None => panic!("Illegal queen move dx {} dy {}", x_diff, y_diff),
        },
    };

    (from.x() as usize, from.y() as usize, move_index)
}
