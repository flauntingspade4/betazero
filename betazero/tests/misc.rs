// use crate::Position;
use citron_core::{move_gen::Move, piece::PieceKind, PlayableTeam, Position};

// todo probably some more tests
#[test]
fn probability_index() {
    let test_move = Move::new(
        Position::new(4, 0),
        Position::new(2, 2),
        PieceKind::Bishop,
        PieceKind::None,
    );
    assert_eq!(
        move_to_probability_index(&test_move, PlayableTeam::White),
        (4, 0, 50)
    );
}
