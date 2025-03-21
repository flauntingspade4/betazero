use std::{
    fs::{read_to_string, File},
    io::BufWriter,
};

use ndarray::Array3;

use citron_core::{nn_util::board_to_network_input, Board, Team};
use pgnparse::parser::parse_pgn_to_rust_struct;
use serde::{Deserialize, Serialize};
use serde_pickle::SerOptions;

pub fn prepare_games() {
    let input = read_to_string(r"games\CCRL.40-2.Archive.[2165313].pgn").unwrap();
    let mut input = input.as_str();

    let mut records = Vec::new();

    for i in 0..500 {
        let mut output = parse_pgn_to_rust_struct(input);
        let won_team = match output.get_header("Result").as_str() {
            "1-0" => Team::White,
            "0-1" => Team::Black,
            "1/2-1/2" => Team::Neither,
            a => {
                println!("Incorrect victory team {a}");
                break;
            }
        };
        let starting_length = records.len();
        for played_move in &output.moves {
            let board = Board::from_fen(&played_move.fen_after).unwrap();
            let record = AERecord {
                result: match won_team.compare(&(!Into::<Team>::into(board.to_play()))) {
                    citron_core::TeamComparison::Same => [1., 0., 0.],
                    citron_core::TeamComparison::Different => [0., 0., 1.],
                    citron_core::TeamComparison::None => [0., 1., 0.],
                },
                board: board_to_network_input(&board, board.to_play()),
            };

            records.push(record);
        }
        println!(
            "({i}) Added {} positions for a total of {} positions ({} won)",
            records.len() - starting_length,
            records.len(),
            won_team
        );

        if let Some(next_game_index) = input.find("\n[Event") {
            input = &input[1 + next_game_index..];
        } else {
            break;
        }
    }

    let f = File::create("latest2.pickle").unwrap();
    let mut f = BufWriter::new(f);
    serde_pickle::to_writer(&mut f, &records, SerOptions::new()).unwrap();
}

/*struct GameReader {
    current_game: Game,
    current_record: Vec<Array3<u64>>,
    result: [f32; 3],
}

impl Visitor for GameReader {
    type Result = Vec<AERecord>;

    fn begin_game(&mut self) {
        self.current_game = Game::new();
        self.result = [0., 0., 0.];
    }

    fn san(&mut self, san: SanPlus) {
        match san.san {
            pgn_reader::San::Normal {
                role,
                file,
                rank,
                capture,
                to,
                promotion,
            } => {
                println!("{:?} from {:?} {:?} to {}", role, file, rank, to);
                let from = Position::new(
                    match file.unwrap() {
                        pgn_reader::File::A => 0,
                        pgn_reader::File::B => 1,
                        pgn_reader::File::C => 2,
                        pgn_reader::File::D => 3,
                        pgn_reader::File::E => 4,
                        pgn_reader::File::F => 5,
                        pgn_reader::File::G => 6,
                        pgn_reader::File::H => 7,
                    },
                    match rank.unwrap() {
                        pgn_reader::Rank::First => 0,
                        pgn_reader::Rank::Second => 1,
                        pgn_reader::Rank::Third => 2,
                        pgn_reader::Rank::Fourth => 3,
                        pgn_reader::Rank::Fifth => 4,
                        pgn_reader::Rank::Sixth => 5,
                        pgn_reader::Rank::Seventh => 6,
                        pgn_reader::Rank::Eighth => 7,
                    },
                );
                let to = square_to_position(to);

                self.current_game = self.current_game.make_move(&Move::new(
                    from,
                    to,
                    role_to_position(role),
                    self.current_game.board.piece_at(to).kind(),
                ))
            }
            pgn_reader::San::Castle(castling_side) => match castling_side {
                pgn_reader::CastlingSide::KingSide => {
                    self.current_game = self
                        .current_game
                        .castle(citron_core::CastlingSide::KingSide)
                }
                pgn_reader::CastlingSide::QueenSide => {
                    self.current_game = self
                        .current_game
                        .castle(citron_core::CastlingSide::KingSide)
                }
            },
            pgn_reader::San::Put { role, to } => {
                println!("Putting {:?} {}", role, to);
                // let added_piece = Piece::new(self.current_game.to_play(), role_to_position(role));
                // self.current_game
                // .add_piece(added_piece, square_to_position(to));
            }
            pgn_reader::San::Null => todo!(),
        }
    }

    fn outcome(&mut self, outcome: Option<pgn_reader::Outcome>) {
        self.result = if let Some(outcome) = outcome {
            match outcome {
                pgn_reader::Outcome::Decisive { winner } => match winner {
                    pgn_reader::Color::Black => [0., 0., 1.],
                    pgn_reader::Color::White => [1., 0., 0.],
                },
                pgn_reader::Outcome::Draw => [0., 1., 0.],
            }
        } else {
            panic!("What the fridgeballs?")
        }
    }

    fn end_game(&mut self) -> Self::Result {
        let mut output = Vec::with_capacity(self.current_record.len());
        output.append(&mut self.current_record);

        output
            .into_iter()
            .map(|r| AERecord {
                board: r,
                result: self.result,
            })
            .collect()
    }

    fn begin_variation(&mut self) -> pgn_reader::Skip {
        Skip(true)
    }
}

fn role_to_position(role: Role) -> PieceKind {
    match role {
        Role::Pawn => PieceKind::Pawn,
        Role::Knight => PieceKind::Knight,
        Role::Bishop => PieceKind::Bishop,
        Role::Rook => PieceKind::Rook,
        Role::Queen => PieceKind::Queen,
        Role::King => PieceKind::King,
    }
}

fn square_to_position(square: Square) -> Position {
    match square {
        Square::A1 => Position::new(0, 0),
        Square::B1 => Position::new(1, 0),
        Square::C1 => Position::new(2, 0),
        Square::D1 => Position::new(3, 0),
        Square::E1 => Position::new(4, 0),
        Square::F1 => Position::new(5, 0),
        Square::G1 => Position::new(6, 0),
        Square::H1 => Position::new(7, 0),
        Square::A2 => Position::new(0, 1),
        Square::B2 => Position::new(1, 1),
        Square::C2 => Position::new(2, 1),
        Square::D2 => Position::new(3, 1),
        Square::E2 => Position::new(4, 1),
        Square::F2 => Position::new(5, 1),
        Square::G2 => Position::new(6, 1),
        Square::H2 => Position::new(7, 1),
        Square::A3 => Position::new(0, 2),
        Square::B3 => Position::new(1, 2),
        Square::C3 => Position::new(2, 2),
        Square::D3 => Position::new(3, 2),
        Square::E3 => Position::new(4, 2),
        Square::F3 => Position::new(5, 2),
        Square::G3 => Position::new(6, 2),
        Square::H3 => Position::new(7, 2),
        Square::A4 => Position::new(0, 3),
        Square::B4 => Position::new(1, 3),
        Square::C4 => Position::new(2, 3),
        Square::D4 => Position::new(3, 3),
        Square::E4 => Position::new(4, 3),
        Square::F4 => Position::new(5, 3),
        Square::G4 => Position::new(6, 3),
        Square::H4 => Position::new(7, 3),
        Square::A5 => Position::new(0, 4),
        Square::B5 => Position::new(1, 4),
        Square::C5 => Position::new(2, 4),
        Square::D5 => Position::new(3, 4),
        Square::E5 => Position::new(4, 4),
        Square::F5 => Position::new(5, 4),
        Square::G5 => Position::new(6, 4),
        Square::H5 => Position::new(7, 4),
        Square::A6 => Position::new(0, 5),
        Square::B6 => Position::new(1, 5),
        Square::C6 => Position::new(2, 5),
        Square::D6 => Position::new(3, 5),
        Square::E6 => Position::new(4, 5),
        Square::F6 => Position::new(5, 5),
        Square::G6 => Position::new(6, 5),
        Square::H6 => Position::new(7, 5),
        Square::A7 => Position::new(0, 6),
        Square::B7 => Position::new(1, 6),
        Square::C7 => Position::new(2, 6),
        Square::D7 => Position::new(3, 6),
        Square::E7 => Position::new(4, 6),
        Square::F7 => Position::new(5, 6),
        Square::G7 => Position::new(6, 6),
        Square::H7 => Position::new(7, 6),
        Square::A8 => Position::new(0, 7),
        Square::B8 => Position::new(1, 7),
        Square::C8 => Position::new(2, 7),
        Square::D8 => Position::new(3, 7),
        Square::E8 => Position::new(4, 7),
        Square::F8 => Position::new(5, 7),
        Square::G8 => Position::new(6, 7),
        Square::H8 => Position::new(7, 7),
    }
}*/

#[derive(Serialize, Deserialize)]
pub struct AERecord {
    board: Array3<u64>,
    result: [f32; 3],
}
