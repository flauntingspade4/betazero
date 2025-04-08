use citron_core::{move_gen::Move, piece::PieceKind, Board, PlayableTeam, Position};
use ndarray::{Array4, Axis};
use std::sync::LazyLock;

pub fn network_input_to_board(input: &Array4<f32>, played_team: PlayableTeam) -> Board {
    let mut board = Board::EMPTY_BOARD;
    use citron_core::piece::Piece::*;

    for (xi, x) in input.axis_iter(Axis(1)).enumerate() {
        // println!("xi {}, X = {:?}", xi, x);
        for (yi, y) in x.axis_iter(Axis(1)).enumerate() {
            // println!("yi {}, Y = {:?}", yi, y);
            for (p_index, present) in y.columns().into_iter().enumerate() {
                // println!("Now on p_index {:?}", p_index);
                if present.get(0) == Some(&1.) {
                    let true_p_index = if played_team == PlayableTeam::Black {
                        if p_index > 5 {
                            p_index - 6
                        } else {
                            p_index + 6
                        }
                    } else {
                        p_index
                    };
                    let piece = match true_p_index {
                        0 => WhitePawn,
                        1 => WhiteRook,
                        2 => WhiteKnight,
                        3 => WhiteBishop,
                        4 => WhiteQueen,
                        5 => WhiteKing,
                        6 => BlackPawn,
                        7 => BlackRook,
                        8 => BlackKnight,
                        9 => BlackBishop,
                        10 => BlackQueen,
                        11 => BlackKing,
                        _ => unreachable!(),
                    };
                    if played_team == PlayableTeam::White {
                        board.add_piece(piece, Position::new(xi as u8, yi as u8));
                    } else {
                        board.add_piece(piece, Position::new(7 - xi as u8, 7 - yi as u8));
                    }
                }
            }
        }
    }

    board
}

#[test]
fn network_input_to_board_test() {
    use citron_core::nn::board_to_network_input;

    let handle = crate::session_handle::BZSessionHandle::load(None);
    let board = Board::from_fen("b3k2b/8/8/3n4/1B4n1/2PP4/4PPP1/4K2B w - - 0 1").unwrap();
    let network_input = board_to_network_input(&board, PlayableTeam::White);
    let network_output = handle
        .call(tensorflow::Tensor::from(network_input.clone()))
        .unwrap();

    println!("{}\n{}", board, network_output.1);
    let new_board =
        network_input_to_board(&network_input.insert_axis(Axis(0)), PlayableTeam::White);
    let network_input = board_to_network_input(&new_board, PlayableTeam::Black);

    let network_output = handle
        .call(tensorflow::Tensor::from(network_input))
        .unwrap();

    println!("Reconstructed board\n{}\n{}", new_board, network_output.1);
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
pub fn move_to_probability_index(item: &Move, team: PlayableTeam) -> (usize, usize, usize) {
    let (from, to) = item.from_to();

    let (x_diff, y_diff) = match team {
        PlayableTeam::White => (
            to.x() as i16 - from.x() as i16,
            to.y() as i16 - from.y() as i16,
        ),
        PlayableTeam::Black => (
            from.x() as i16 - to.x() as i16,
            from.y() as i16 - to.y() as i16,
        ),
    };

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

    match team {
        PlayableTeam::White => (from.x() as usize, from.y() as usize, move_index),
        PlayableTeam::Black => (7 - from.x() as usize, 7 - from.y() as usize, move_index),
    }
}

#[test]
fn flip_probability_index() {
    let w_move = Move::new(
        Position::new(0, 0),
        Position::new(1, 2),
        PieceKind::Knight,
        PieceKind::None,
    );
    let b_move = Move::new(
        Position::new(7, 7),
        Position::new(6, 5),
        PieceKind::Knight,
        PieceKind::None,
    );

    assert_eq!(
        move_to_probability_index(&w_move, PlayableTeam::White),
        move_to_probability_index(&b_move, PlayableTeam::Black)
    );

    let w_move = Move::new(
        Position::new(2, 1),
        Position::new(6, 5),
        PieceKind::Bishop,
        PieceKind::None,
    );
    let b_move = Move::new(
        Position::new(5, 6),
        Position::new(1, 2),
        PieceKind::Bishop,
        PieceKind::None,
    );

    assert_eq!(
        move_to_probability_index(&w_move, PlayableTeam::White),
        move_to_probability_index(&b_move, PlayableTeam::Black)
    );

    let w_move = Move::new(
        Position::new(4, 1),
        Position::new(4, 3),
        PieceKind::Pawn,
        PieceKind::None,
    );
    let b_move = Move::new(
        Position::new(3, 6),
        Position::new(3, 4),
        PieceKind::Pawn,
        PieceKind::None,
    );

    assert_eq!(
        move_to_probability_index(&w_move, PlayableTeam::White),
        move_to_probability_index(&b_move, PlayableTeam::Black)
    )
}

#[test]
fn flip_probability_from_network() {
    use citron_core::{nn::board_to_network_input, Board};
    let handle = crate::session_handle::BZSessionHandle::load(None);

    let board = &Board::from_fen("k7/2n5/1n6/3r4/4R3/6N1/5N2/7K w - - 0 1").unwrap();
    let board_input = board_to_network_input(&board, PlayableTeam::White);
    let output = handle.call(tensorflow::Tensor::from(&board_input)).unwrap();

    let flipped_board_input = board_to_network_input(&board, PlayableTeam::Black);
    let flipped_output = handle
        .call(tensorflow::Tensor::from(flipped_board_input))
        .unwrap();

    assert_eq!(output.0, flipped_output.0);
    assert_eq!(output.1, flipped_output.1);
}
