mod low_level;
mod types;
mod assume;
mod board;
mod neighbors;

use crate::board::*;
use crate::types::*;
use clap::Parser;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

/// Sudoku solver
#[derive(Parser, Debug)]
struct Args {
    /// Print the unsolved and solved boards
    #[structopt(short, long)]
    print: bool,

    /// Check solved boards for correctness
    #[structopt(short, long)]
    verify: bool,

    /// Number of benchmark rounds to run
    #[structopt(short, long, default_value = "1")]
    rounds: u32,

    /// File argument with one board per line
    #[structopt(long, default_value = "")]
    boards_file: String,

    /// Sudoku board argument
    #[structopt(long, default_value = "")]
    board: String,

    /// Whether to solve boards in parallel
    #[structopt(long)]
    parallel: bool,
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn do_board(args: &Args, line: &str) {
    let o = parse_board(line);
    if let Some(mut board) = o {
        if args.print {
            println!("\nUnsolved:\n");
            print!("{board}");
            println!("\n---------------------\n");
        }

        let solved = solve(&mut board);

        if let Some(solved_board) = solved {
            if args.print {
                print!("{solved_board}");
            }
            if args.verify && !solved_board.is_complete() {
                panic!("Invalid solved board!");
            }
        } else {
            println!("Not Solved!");
            if args.verify {
                panic!("Unsolved board found!")
            }
        }
    }
}

fn main() {
    let args = Args::parse();

    for _ in 0..args.rounds {
        if !args.board.is_empty() {
            do_board(&args, &args.board);
        } else if let Ok(lines) = read_lines(&args.boards_file) {
            if args.parallel {
                lines
                    .par_bridge()
                    .for_each(|r| do_board(&args, &r.unwrap()));
            } else {
                lines.for_each(|r| do_board(&args, &r.unwrap()));
            }
        } else {
            panic!("Must provide either --boards_file or --board !")
        }
    }
}
