from Block import Block


class Player(Block):
    def __init__(self, path, screen_size, offset, x_pos, y_pos, speed):
        super().__init__(path, x_pos, y_pos)
        self.screen_width, self.screen_height = screen_size
        self.speed = speed
        self.offset = offset
        self.movement = 0

    def screen_constrain(self):
        if self.rect.top <= 0 + self.offset:
            self.rect.top = 0 + self.offset
        if self.rect.bottom >= self.screen_height - self.offset:
            self.rect.bottom = self.screen_height - self.offset

    def update(self, ball_group):
        self.rect.y += self.movement
        self.screen_constrain()
