# FRL-Distributed-ML-Scaffold
FRL Distributed ML Scaffold is a set of training scripts intended to simplify defining, training, and debugging a multi-task machine learning problem. Problems implemented on this framework get out-of-the-box distributed training and multithreaded online data preprocessing support.

## Requirements
FRL Distributed ML Scaffold requires or works with
* Mac OS X or Linux

## Getting Started with FRL Distributed ML Scaffold
To get started, run `setup.py install`.
Set up a problem by inheriting from and implementing the API from the `Problem` class in `problem.py`.
The runner for problems is `Solver.solve()`.

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
FRL Distributed ML Scaffold is MIT licensed, as found in the LICENSE file.
