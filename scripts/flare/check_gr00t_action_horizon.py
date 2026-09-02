from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


def main() -> None:
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path="checkpoints/GR00T-N1.7-LIBERO/libero_10",
        device="cuda",
    )

    action_head = policy.model.action_head

    print("config action_horizon:", action_head.config.action_horizon)
    print("internal action_dim:", action_head.action_dim)


if __name__ == "__main__":
    main()