# How to contribute

We would love any contributions, in code and otherwise. Whether its a how-to-use problem, a bug report, or idea for enhancement, please check if an *issue* for the topic already exists, and if not, open one yourself and clearly describe what is in your mind. If you are willing to write code, that is also a good place for us to agree on how to proceed. You don't have to be genius with years of experience; we are all learning. :]

## Getting your code in

TL;DR: Create issue, fork, write code in a branch, PR

### Getting development environment (in Linux/OSX)

1. Fork this repository under your own account.
1. Clone the forked repository to your computer and cd in.
1. Install with: python setup.py develop

### Using Git 

Inside the cloned project directory, you can do following to ensure you are up to date and then create the branch for contribution:
1. Add remote for upstream updates (only once): git remote add upstream git@github.com:Teekuningas/icasso.git
1. Download and merge updates from upstream: git pull upstream master
1. Give a good name for feature / bugfix branch and switch to it: git checkout -b fix-bad-code

[//]: # (Hello)

Now you are all set to write your code.

### Writing code

Try to keep your code clean. Use for example pycodestyle or flake8 to check your pep8-compliance. Generated code from Qtdesigner is kept as it is.

### Finishing with pull request

When finished, you should save your changes (if not done already):
1. Save changes: git commit
1. Push changes to your fork: git push origin fix-bad-code
1. Go to your forked repository on github and it should happily notify you about the push you have just made and allow you to click *Compare & pull request*. Fill the details and let the code fly.

## License

When contributing, your code is licensed under MIT license.
