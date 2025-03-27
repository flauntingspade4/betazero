use std::{
    fs::{read_to_string, File},
    io::BufWriter,
};

use ndarray::Array3;

use citron_core::{nn_util::board_to_network_input, Board, Team};
use pgnparse::parser::parse_pgn_to_rust_struct;
use serde::{Deserialize, Serialize};
use serde_pickle::SerOptions;
use tfrecord::{Example, Feature, RecordWriter};

// const POSITIONS_PER_FILE: usize = 10000;

pub fn prepare_games() {
    let input = read_to_string(r"games\CCRL.40-2.Archive.[2165313].pgn").unwrap();
    let mut input = input.as_str();

    println!("Loaded game");

    let mut won = Vec::new();
    let mut lost = Vec::new();

    for i in 0..2 {
        if i % 100 == 0 {
            println!("On game {i}",);
        }

        // let mut output = parse_pgn_to_rust_struct(input);
        let mut output = if let Some(next_game_index) = input.find("\n[Event") {
            let output = parse_pgn_to_rust_struct(&input[..next_game_index]);
            input = &input[1 + next_game_index..];
            output
        } else {
            break;
        };

        let won_team = match output.get_header("Result").as_str() {
            "1-0" => Team::White,
            "0-1" => Team::Black,
            "1/2-1/2" => continue,
            a => {
                println!("Incorrect victory team {a}");
                break;
            }
        };
        for played_move in &output.moves {
            let board = Board::from_fen(&played_move.fen_before).unwrap();
            println!("{board}");
            let r = board_to_network_input(&board, board.to_play());
            println!("{:?}", r);
            println!("{:?}", r.clone().into_raw_vec());
            let record = AERecord { board: r };

            match won_team.compare(&board.to_play().into()) {
                citron_core::TeamComparison::Same => won.push(record),
                citron_core::TeamComparison::Different => lost.push(record),
                citron_core::TeamComparison::None => panic!("How the fridge??"),
            }

            break;
        }
    }
    // if white_won.len() > POSITIONS_PER_FILE {
    //     println!("Writing {} white_won positions", white_won.len());
    //     write_file(&format!(r"white_won\file"), &mut white_won);
    //     white_won.clear();
    // }

    // if white_lost.len() > POSITIONS_PER_FILE {
    //     println!("Writing {} white_lost positions", white_lost.len());
    //     write_file(&format!(r"white_lost\file"), &mut white_lost);
    //     white_lost.clear();
    // }

    // if black_won.len() > POSITIONS_PER_FILE {
    //     println!("Writing {} black_won positions", black_won.len());
    //     write_file(&format!(r"black_won\file"), &mut black_won);
    //     black_won.clear();
    // }

    // if black_lost.len() > POSITIONS_PER_FILE {
    //     println!("Writing {} black_lost positions", black_lost.len());
    //     write_file(&format!(r"black_lost\file"), &mut black_lost);
    //     black_lost.clear();
    // }

    let mut won_writer = RecordWriter::create("won.tfrecord").unwrap();

    for won_position in won {
        let board_feature = Feature::from_f32_list(won_position.board.into_raw_vec());

        let example = vec![("board".into(), board_feature)]
            .into_iter()
            .collect::<Example>();

        won_writer.send(example).unwrap();
    }

    let mut lost_writer = RecordWriter::create("won.tfrecord").unwrap();

    for lostwon_position in lost {
        let board_feature = Feature::from_f32_list(lostwon_position.board.into_raw_vec());

        let example = vec![("board".into(), board_feature)]
            .into_iter()
            .collect::<Example>();

        lost_writer.send(example).unwrap();
    }
}

fn write_file(path: &str, records: &mut Vec<AERecord>) {
    let f = File::create(path).unwrap();
    let mut f = BufWriter::new(f);
    serde_pickle::to_writer(&mut f, &records, SerOptions::new()).unwrap();
}

#[derive(Serialize, Deserialize)]
pub struct AERecord {
    board: Array3<f32>,
}
