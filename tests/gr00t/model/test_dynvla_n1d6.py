import torch

from gr00t.model.modules.dit import AlternateVLDiT
from gr00t.model.modules.dynvla_dynamics import DynVLAAlternateVLDiT


def _model_kwargs():
    return {
        "num_attention_heads": 4,
        "attention_head_dim": 8,
        "output_dim": 16,
        "num_layers": 4,
        "dropout": 0.0,
        "final_dropout": False,
        "positional_embeddings": None,
        "interleave_self_attention": True,
        "cross_attention_dim": 24,
        "attend_text_every_n_blocks": 2,
    }


def test_dynvla_preserves_pretrained_dit_parameter_names():
    base = AlternateVLDiT(**_model_kwargs())
    dynvla = DynVLAAlternateVLDiT(
        **_model_kwargs(),
        constraint_depth=2,
        codebook_size=8,
        codebook_temperature=0.1,
    )

    base_keys = set(base.state_dict())
    dynvla_keys = set(dynvla.state_dict())
    assert base_keys <= dynvla_keys


def test_dynvla_midpoint_retrieval_has_gradients():
    model = DynVLAAlternateVLDiT(
        **_model_kwargs(),
        constraint_depth=2,
        codebook_size=8,
        codebook_temperature=0.1,
        dynamics_teacher_dropout=1.0,
    )
    hidden_states = torch.randn(2, 5, 32)
    encoder_hidden_states = torch.randn(2, 7, 24)
    timestep = torch.tensor([100, 500])
    image_mask = torch.tensor(
        [[True, True, False, False, False, False, False]] * 2,
        dtype=torch.bool,
    )
    backbone_attention_mask = torch.ones(2, 7, dtype=torch.bool)
    dynamics_hint = torch.randn(2, 32)

    output, hidden, auxiliary = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        image_mask=image_mask,
        backbone_attention_mask=backbone_attention_mask,
        dynamics_hint=dynamics_hint,
        return_all_hidden_states=True,
        return_dynvla_aux=True,
    )

    assert output.shape == (2, 5, 16)
    assert len(hidden) == 5
    assert auxiliary["layer_queries"].shape == (2, 2, 32)
    assert auxiliary["retrieved_prototypes"].shape == (2, 2, 32)
    assert auxiliary["bank_weights"].shape == (2, 2, 8)

    output.square().mean().backward()
    assert model.bank.prototypes.grad is not None
    assert torch.isfinite(model.bank.prototypes.grad).all()


def test_dynvla_frozen_bank_keeps_other_dynamics_trainable():
    model = DynVLAAlternateVLDiT(
        **_model_kwargs(),
        constraint_depth=2,
        codebook_size=8,
        codebook_temperature=0.1,
    )
    model.set_dynvla_trainable(True, train_bank=False)

    assert model.identification_token.requires_grad
    assert all(parameter.requires_grad for parameter in model.query_projection.parameters())
    assert not model.bank.prototypes.requires_grad

    output, _, _ = model(
        hidden_states=torch.randn(2, 5, 32),
        encoder_hidden_states=torch.randn(2, 7, 24),
        timestep=torch.tensor([100, 500]),
        image_mask=torch.tensor(
            [[True, True, False, False, False, False, False]] * 2,
            dtype=torch.bool,
        ),
        backbone_attention_mask=torch.ones(2, 7, dtype=torch.bool),
        dynamics_hint=torch.randn(2, 32),
        return_all_hidden_states=True,
        return_dynvla_aux=True,
    )
    output.square().mean().backward()
    assert model.bank.prototypes.grad is None
    assert model.query_projection[1].weight.grad is not None
    assert torch.isfinite(model.query_projection[1].weight.grad).all()


def test_dynvla_random_bank_reset_changes_only_prototypes():
    torch.manual_seed(7)
    source = DynVLAAlternateVLDiT(
        **_model_kwargs(),
        constraint_depth=2,
        codebook_size=8,
        codebook_temperature=0.1,
    )
    target = DynVLAAlternateVLDiT(
        **_model_kwargs(),
        constraint_depth=2,
        codebook_size=8,
        codebook_temperature=0.1,
    )
    target.load_state_dict(source.state_dict())
    inherited = {name: tensor.clone() for name, tensor in target.state_dict().items()}

    target.bank.reset_parameters()
    reset_state = target.state_dict()

    assert not torch.equal(reset_state["bank.prototypes"], inherited["bank.prototypes"])
    for name, tensor in inherited.items():
        if name != "bank.prototypes":
            assert torch.equal(reset_state[name], tensor)
    assert torch.isfinite(reset_state["bank.prototypes"]).all()


def test_dynvla_disabled_bank_bypasses_retrieval_and_stays_finite():
    model = DynVLAAlternateVLDiT(
        **_model_kwargs(),
        constraint_depth=2,
        codebook_size=8,
        codebook_temperature=0.1,
        dynamics_teacher_dropout=1.0,
        disable_dynamics_bank=True,
    )
    model.set_dynvla_trainable(True, train_bank=True)
    hidden_states = torch.randn(2, 5, 32)
    encoder_hidden_states = torch.randn(2, 7, 24)
    timestep = torch.tensor([100, 500])
    image_mask = torch.tensor(
        [[True, True, False, False, False, False, False]] * 2,
        dtype=torch.bool,
    )
    backbone_attention_mask = torch.ones(2, 7, dtype=torch.bool)
    dynamics_hint = torch.randn(2, 32)

    output, hidden, auxiliary = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        image_mask=image_mask,
        backbone_attention_mask=backbone_attention_mask,
        dynamics_hint=dynamics_hint,
        return_all_hidden_states=True,
        return_dynvla_aux=True,
    )

    assert output.shape == (2, 5, 16)
    assert torch.isfinite(output).all()
    assert hidden[2].shape == (2, 7, 32)
    assert auxiliary["retrieved_prototypes"].shape == (2, 2, 32)
    assert torch.equal(
        auxiliary["retrieved_prototypes"], auxiliary["layer_queries"]
    )
    assert auxiliary["bank_weights"] is None
    assert not model.bank.prototypes.requires_grad

    with torch.no_grad():
        model.bank.prototypes.fill_(123.0)
        changed_bank_output, _, _ = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
            dynamics_hint=dynamics_hint,
            return_all_hidden_states=True,
            return_dynvla_aux=True,
        )
    assert torch.equal(output, changed_bank_output)

    output.square().mean().backward()
    assert model.bank.prototypes.grad is None
    assert model.query_projection[1].weight.grad is not None
    assert torch.isfinite(model.query_projection[1].weight.grad).all()
