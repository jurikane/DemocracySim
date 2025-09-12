import mesa


class StubVoteAgent(mesa.Agent):
    """Only the fields the browser portrayal uses."""
    def __init__(self, uid, model, pos, personality_idx):
        super().__init__(uid, model)
        self.pos = pos
        self.personality_idx = personality_idx        # or what you draw


class StubArea(mesa.Agent):
    """Drawn as a rectangle – lives once in the south-west corner cell."""
    def __init__(self, uid, model, pos, h, w):
        super().__init__(uid, model)
        self.pos = pos
        self.height = h
        self.width  = w
