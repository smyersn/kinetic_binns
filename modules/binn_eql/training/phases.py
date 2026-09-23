"""
Four-phase training schedule.

    Phase 1  surface   fit u(x, t) to the data                 loss: GLS
    Phase 2  physics   fit D and F on the frozen surface       loss: PDE
    Phase 3  sparsify  ramp the L0 penalty in linearly         loss: PDE + ramped L0
    Phase 4  prune     full L0; model selection and early      loss: PDE + L0
                       stopping on validation loss

Phases differ only in which sub-networks train, how the loss terms are
weighted, and whether the LR scheduler steps.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    number: int
    trains: tuple          # sub-networks updated: 'surface', 'reaction', 'diffusion'
    gls_weight: float
    pde_weight: float
    l0: str                # 'off', 'ramp' or 'full'
    steps_scheduler: bool
    mass_active: bool


PHASES = {p.number: p for p in (
    Phase(1, ('surface',),              1.0, 0.0, 'off',  steps_scheduler=True,  mass_active=False),
    Phase(2, ('reaction', 'diffusion'), 0.0, 1.0, 'off',  steps_scheduler=False, mass_active=True),
    Phase(3, ('reaction', 'diffusion'), 0.0, 1.0, 'ramp', steps_scheduler=False, mass_active=True),
    Phase(4, ('reaction', 'diffusion'), 0.0, 1.0, 'full', steps_scheduler=False, mass_active=True),
)}


@dataclass
class PhaseSchedule:
    """Epochs at which Phases 1-3 end. Phase 1 early stopping may move phase_1_end earlier."""
    phase_1_end: int
    phase_2_end: int
    phase_3_end: int

    @classmethod
    def default(cls, epochs):
        return cls(int(0.05 * epochs), int(0.2 * epochs), int(0.3 * epochs))

    def validate(self, epochs):
        if not 0 < self.phase_1_end < self.phase_2_end < self.phase_3_end <= epochs:
            raise ValueError(
                f"Phase ends must be strictly increasing and within epochs: got "
                f"({self.phase_1_end}, {self.phase_2_end}, {self.phase_3_end}) with epochs={epochs}")

    def phase_at(self, epoch):
        if epoch < self.phase_1_end:
            return PHASES[1]
        if epoch < self.phase_2_end:
            return PHASES[2]
        if epoch < self.phase_3_end:
            return PHASES[3]
        return PHASES[4]

    def l0_weight(self, phase, epoch, l0_weight, l0_scale):
        """Effective L0 weight l0_weight * l0_scale, ramped linearly from 0 across Phase 3."""
        if phase.l0 == 'off':
            return 0.0
        full = l0_weight * l0_scale
        if phase.l0 == 'full':
            return full
        progress = (epoch - self.phase_2_end) / max(self.phase_3_end - self.phase_2_end, 1)
        return full * progress