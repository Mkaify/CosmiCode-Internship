"""
Tic-Tac-Toe Game (Player vs Player)

A simple implementation of the classic Tic-Tac-Toe game where two human players
take turns playing against each other on a 3x3 grid.

Features:
- Clean console interface
- Input validation
- Win/draw detection
- Score tracking
- Multiple game rounds
"""

class TicTacToe:
    def __init__(self):
        """Initialize the game board and players"""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
        
    def print_board(self):
        """Display the current game board"""
        print("\n   0   1   2")
        print("  -----------")
        for i in range(3):
            print(f"{i}| {self.board[i][0]} | {self.board[i][1]} | {self.board[i][2]} |")
            print("  -----------")
        print()
    
    def is_valid_move(self, row, col):
        """
        Check if a move is valid
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            
        Returns:
            bool: True if move is valid, False otherwise
        """
        if row < 0 or row > 2 or col < 0 or col > 2:
            return False
        return self.board[row][col] == ' '
    
    def make_move(self, row, col):
        """
        Make a move on the board
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            
        Returns:
            bool: True if move was successful, False otherwise
        """
        if self.is_valid_move(row, col):
            self.board[row][col] = self.current_player
            return True
        return False
    
    def check_winner(self):
        """
        Check if there's a winner or if the game is a draw
        
        Returns:
            str: 'X', 'O', 'Draw', or None
        """
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        
        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        # Check for draw
        if all(self.board[i][j] != ' ' for i in range(3) for j in range(3)):
            return 'Draw'
        
        return None
    
    def switch_player(self):
        """Switch the current player"""
        self.current_player = 'O' if self.current_player == 'X' else 'X'
    
    def reset_game(self):
        """Reset the game for a new round"""
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'
        self.game_over = False
        self.winner = None
    
    def get_player_input(self):
        """
        Get valid input from the current player
        
        Returns:
            tuple: (row, col) coordinates
        """
        while True:
            try:
                print(f"Player {self.current_player}'s turn")
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter column (0-2): "))
                
                if self.is_valid_move(row, col):
                    return row, col
                else:
                    print("Invalid move! That position is already taken or out of bounds.")
                    print("Please try again.\n")
                    
            except ValueError:
                print("Invalid input! Please enter numbers only.")
                print("Please try again.\n")
            except KeyboardInterrupt:
                print("\nGame interrupted!")
                return None, None
    
    def play_round(self):
        """
        Play a single round of Tic-Tac-Toe
        
        Returns:
            str: Winner ('X', 'O', 'Draw') or None if game was interrupted
        """
        self.print_board()
        
        while not self.game_over:
            # Get player input
            row, col = self.get_player_input()
            
            # Handle game interruption
            if row is None:
                return None
            
            # Make the move
            if self.make_move(row, col):
                self.print_board()
                
                # Check for winner
                result = self.check_winner()
                if result:
                    self.winner = result
                    self.game_over = True
                    
                    if result == 'Draw':
                        print("It's a draw! 🤝")
                    else:
                        print(f"🎉 Player {result} wins! 🎉")
                    
                    return result
                
                # Switch players
                self.switch_player()
            else:
                print("Something went wrong. Please try again.")
        
        return self.winner

def print_game_instructions():
    """Print game instructions"""
    print("=" * 50)
    print("         WELCOME TO TIC-TAC-TOE!")
    print("=" * 50)
    print("Instructions:")
    print("• This is a 2-player game (Player X vs Player O)")
    print("• Players take turns placing their marks")
    print("• Enter row and column coordinates (0-2)")
    print("• First player to get 3 in a row wins!")
    print("• Rows, columns, or diagonals all count")
    print("=" * 50)

def play_tournament():
    """
    Play multiple rounds with score tracking
    """
    print_game_instructions()
    
    game = TicTacToe()
    scores = {'X': 0, 'O': 0, 'Draw': 0}
    round_number = 1
    
    while True:
        print(f"\n🎯 ROUND {round_number} 🎯")
        print(f"Score - Player X: {scores['X']}, Player O: {scores['O']}, Draws: {scores['Draw']}")
        
        # Reset game for new round
        game.reset_game()
        
        # Play the round
        result = game.play_round()
        
        # Handle game interruption
        if result is None:
            print("Thanks for playing!")
            break
        
        # Update scores
        scores[result] += 1
        round_number += 1
        
        # Ask if players want to continue
        while True:
            try:
                continue_game = input("\nDo you want to play another round? (y/n): ").lower().strip()
                if continue_game in ['y', 'yes']:
                    break
                elif continue_game in ['n', 'no']:
                    print("\n🏆 FINAL SCORES 🏆")
                    print("=" * 30)
                    print(f"Player X wins: {scores['X']}")
                    print(f"Player O wins: {scores['O']}")
                    print(f"Draws: {scores['Draw']}")
                    
                    if scores['X'] > scores['O']:
                        print("🎉 Player X is the overall winner! 🎉")
                    elif scores['O'] > scores['X']:
                        print("🎉 Player O is the overall winner! 🎉")
                    else:
                        print("🤝 It's a tie overall! 🤝")
                    
                    print("Thanks for playing!")
                    return
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            except KeyboardInterrupt:
                print("\nThanks for playing!")
                return

def demo_game():
    """Demonstrate a quick game"""
    print("Quick Demo Game:")
    print("=" * 20)
    
    game = TicTacToe()
    
    # Demo moves
    demo_moves = [
        (0, 0), (0, 1), (1, 1), (0, 2), (2, 2)
    ]
    
    for i, (row, col) in enumerate(demo_moves):
        print(f"Player {game.current_player} plays at ({row}, {col})")
        game.make_move(row, col)
        game.print_board()
        
        result = game.check_winner()
        if result:
            if result == 'Draw':
                print("Demo ended in a draw!")
            else:
                print(f"Demo: Player {result} wins!")
            break
        
        game.switch_player()

if __name__ == "__main__":
    try:
        choice = input("Choose an option:\n"
                      "1. Play tournament\n"
                      "2. Quick demo\n"
                      "Enter your choice (1-2): ").strip()
        
        if choice == '1':
            play_tournament()
        elif choice == '2':
            demo_game()
        else:
            print("Invalid choice. Starting tournament mode...")
            play_tournament()
            
    except KeyboardInterrupt:
        print("\nThanks for playing Tic-Tac-Toe!") 