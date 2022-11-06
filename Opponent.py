from Block import Block


class Opponent(Block):
    def __init__(self, path, screen_size, offset, x_pos, y_pos, speed):
        super().__init__(path, x_pos, y_pos)
        self.screen_width, self.screen_height = screen_size
        self.speed = speed
        self.offset = offset

    def screen_constrain(self):
        if self.rect.top <= 0 + self.offset:
            self.rect.top = 0 + self.offset
        if self.rect.bottom >= self.screen_height - self.offset:
            self.rect.bottom = self.screen_height - self.offset

    def update(self, ball_group):
        if self.rect.top < ball_group.sprite.rect.y:
            self.rect.y += self.speed
        if self.rect.bottom > ball_group.sprite.rect.y:
            self.rect.y -= self.speed
        self.screen_constrain()
