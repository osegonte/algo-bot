class MartingaleStrategy:
    def __init__(self, initial_bet=100, multiplier=2):
        self.initial_bet = initial_bet
        self.multiplier = multiplier
        self.current_bet = initial_bet
        
    def next_bet_size(self, won_last=False):
        if won_last:
            self.current_bet = self.initial_bet
        else:
            self.current_bet *= self.multiplier
        return self.current_bet
