use std::{
    fs::{read_to_string, File},
    io::BufWriter,
};

use ndarray::Array3;

use citron_core::{
    nn::{add_move_information_to_nn_input, board_to_network_input},
    Board, PlayableTeam, Team,
};
use pgnparse::parser::parse_pgn_to_rust_struct;
use rand::seq::{IndexedRandom, IteratorRandom};
use serde::{Deserialize, Serialize};
use serde_pickle::SerOptions;

// const POSITIONS_PER_FILE: usize = 10000;

pub fn prepare_games() {
    let input = read_to_string(r"games\CCRL.40-2.Archive.[2165313].pgn").unwrap();
    let mut input = input.as_str();
    let mut rng = rand::rng();

    // println!("Loaded game");

    let mut white_won = Vec::new();
    let mut black_won = Vec::new();

    for i in 0..1_000_000 {
        if i % 100 == 0 {
            println!("On game {i}");
        }

        let mut output = if let Some(next_game_index) = input.find("\n[Event") {
            let output = parse_pgn_to_rust_struct(&input[..next_game_index]);
            input = &input[1 + next_game_index..];
            output
        } else {
            break;
        };

        let (won_team, skip) = match output.get_header("Result").as_str() {
            "1-0" => (PlayableTeam::White, 0),
            "0-1" => (PlayableTeam::Black, 1),
            "1/2-1/2" => continue,
            a => {
                println!("Incorrect victory team {a}");
                break;
            }
        };

        for played_move in output
            .moves
            .iter()
            // .skip(skip)
            // .step_by(2)
            .choose_multiple(&mut rng, 2)
        {
            let board = Board::from_fen(&played_move.fen_before).unwrap();
            // println!(
            //     "{won_team} won. Adding board from fen \"{}\"\n{}",
            //     &played_move.fen_before, board
            // );
            // println!("{board}");
            let mut r = board_to_network_input(&board, PlayableTeam::White);
            add_move_information_to_nn_input(&board, &mut r);
            // println!("{:?}", r);
            // println!("{:?}", r.clone().into_raw_vec());
            // ඞ ඞ ඞ

            match won_team {
                PlayableTeam::White => white_won.push(r),
                PlayableTeam::Black => black_won.push(r),
            }

            // break;
        }
    }
    write_file("white_won_teams.pickle", &mut white_won);
    write_file("black_won_teams.pickle", &mut black_won);
    //     println!("Writing {} black_lost positions", black_lost.len());
    //     write_file(&format!(r"black_lost\file"), &mut black_lost);
    //     black_lost.clear();
    // }

    // let mut won_writer = RecordWriter::create("won.tfrecord").unwrap();

    // for won_position in white_won {
    //     let board_feature = Feature::from_f32_list(won_position.into_raw_vec());

    //     let example = vec![("board".into(), board_feature)]
    //         .into_iter()
    //         .collect::<Example>();

    //     won_writer.send(example).unwrap();
    // }

    // let mut lost_writer = RecordWriter::create("won.tfrecord").unwrap();

    // for lost_position in black_won {
    //     let board_feature = Feature::from_f32_list(lost_position.into_raw_vec());

    //     let example = vec![("board".into(), board_feature)]
    //         .into_iter()
    //         .collect::<Example>();

    //     lost_writer.send(example).unwrap();
    // }
}

fn write_file(path: &str, records: &mut Vec<Array3<f32>>) {
    let f = File::create(path).unwrap();
    println!("Writing {} records to {}", records.len(), path);
    let mut f = BufWriter::new(f);
    serde_pickle::to_writer(&mut f, &records, SerOptions::new()).unwrap();
}

#[derive(Serialize, Deserialize)]
pub struct AERecord {
    board: Array3<f32>,
}
