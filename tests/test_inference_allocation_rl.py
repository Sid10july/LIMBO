import unittest
import tempfile

from src.controllers.inference_allocation_rl import (
    MaskedDQNConfig,
    MaskedDQNInferenceController,
    OS_RL_ACTIONS,
    OS_RL_FEATURE_NAMES,
    RLEpisodeTransition,
    _EpisodeState,
    append_memory_hint_to_pending_user,
    compute_final_reward,
    run_os_inference_rl_episode,
    run_self_critique_verifier,
    torch,
    valid_os_rl_action_mask,
)
from src.agents import Agent
from src.callbacks import CallbackHandler
from src.tasks import DatasetItem, Task
from src.typings import ChatHistoryItem, Role, SampleStatus, Session, TaskName


class _FakeAgent:
    def get_role_dict(self):
        return {Role.USER: "user", Role.AGENT: "assistant"}


class _NoLanguageModelAgent:
    pass


class _FakeRetrievalCallback:
    def __init__(self, memory_session):
        self.memory_session = memory_session

    def _select_retrieved_sessions(self, **kwargs):
        return [self.memory_session]

    def _render_session_block(self, session, agent_role_dict):
        return "Question prior task:\nassistant: Act: finish\n"


class _TinyDatasetItem(DatasetItem):
    instruction: str = "Create /tmp/example."
    skill_list: list[str] = ["touch"]

    def get_skill_list(self):
        return self.skill_list

    def get_difficulty_level(self):
        return 0


class _TinyOSTask(Task[_TinyDatasetItem]):
    def __init__(self):
        super().__init__(
            task_name=TaskName.OS_INTERACTION,
            chat_history_item_factory=object(),
            max_round=2,
        )
        self._set_dataset({"0": _TinyDatasetItem()})
        self.tool_budget = 4
        self.stop_enabled = True
        self.container = None

    def _get_default_task_output(self):
        return {"answer": None}

    @staticmethod
    def _parse_agent_response(agent_response):
        from src.tasks.task import AgentAction, AgentResponseParserResult

        if "Act: finish" in agent_response:
            return AgentResponseParserResult(
                action=AgentAction.FINISH,
                content=None,
                finish_reason=None,
            )
        return AgentResponseParserResult(
            action=AgentAction.INVALID,
            content=None,
            finish_reason="invalid",
        )

    def _reset(self, session):
        session.chat_history.inject({"role": Role.USER, "content": "rules"})
        session.chat_history.inject({"role": Role.AGENT, "content": "OK."})
        session.chat_history.inject(
            {"role": Role.USER, "content": "Create /tmp/example."}
        )

    def _interact(self, session):
        session.sample_status = SampleStatus.COMPLETED
        session.task_output = {"answer": None}

    def _complete(self, session):
        from src.typings import SessionEvaluationOutcome

        session.evaluation_record.outcome = SessionEvaluationOutcome.CORRECT

    def _release(self):
        pass

    def calculate_metric(self, session_partial_list):
        return {}


class _FinishAgent(Agent):
    def _inference(self, chat_history):
        return ChatHistoryItem(role=Role.AGENT, content="Act: finish")

    def get_role_dict(self):
        return {Role.USER: "user", Role.AGENT: "assistant"}


class _FakeCostTracker:
    calls = []


class _ScriptedController:
    action_names = OS_RL_ACTIONS
    epsilon = 0.0
    episodes_seen = 0
    training_steps = 0

    def __init__(self, actions):
        self.actions = list(actions)
        self.observed_reward = None

    def select_action(self, state, valid_action_mask):
        action_name = self.actions.pop(0)
        action_idx = OS_RL_ACTIONS.index(action_name)
        assert valid_action_mask[action_idx]
        return action_idx, {"scripted": True}

    def observe_episode(self, transitions, final_reward):
        self.observed_reward = final_reward
        return {"updates": 0, "loss": None, "buffer_size": len(transitions)}

    def save_state(self, path):
        return None


class InferenceAllocationRLTest(unittest.TestCase):
    def test_setup_mask_allows_initial_choices_but_not_retry(self):
        state = _EpisodeState(stage="setup")
        mask = valid_os_rl_action_mask(
            state,
            current_session=None,
            max_attempts=2,
            max_retrieval_injections=1,
            max_verifications=1,
        )
        allowed = {
            action for action, is_allowed in zip(OS_RL_ACTIONS, mask) if is_allowed
        }
        self.assertIn("retrieve_top1", allowed)
        self.assertNotIn("retrieve_top4", allowed)
        self.assertIn("reason_step", allowed)
        self.assertNotIn("terminate", allowed)
        self.assertNotIn("retry", allowed)
        self.assertNotIn("verify", allowed)

    def test_staged_mask_forces_progress_after_control_action(self):
        state = _EpisodeState(stage="running", force_reason_next=True)
        session = Session(task_name=TaskName.OS_INTERACTION, sample_index="0")
        session.sample_status = SampleStatus.RUNNING
        mask = valid_os_rl_action_mask(
            state,
            current_session=session,
            max_attempts=2,
            max_retrieval_injections=2,
            max_verifications=1,
        )
        allowed = {
            action for action, is_allowed in zip(OS_RL_ACTIONS, mask) if is_allowed
        }
        self.assertEqual(allowed, {"reason_step"})

    def test_running_mask_blocks_attempt_terminal_actions(self):
        state = _EpisodeState(stage="running")
        session = Session(task_name=TaskName.OS_INTERACTION, sample_index="0")
        session.sample_status = SampleStatus.RUNNING
        mask = valid_os_rl_action_mask(
            state,
            current_session=session,
            max_attempts=2,
            max_retrieval_injections=2,
            max_verifications=1,
        )
        allowed = {
            action for action, is_allowed in zip(OS_RL_ACTIONS, mask) if is_allowed
        }
        self.assertIn("reason_step", allowed)
        self.assertNotIn("retrieve_top1", allowed)
        self.assertNotIn("retrieve_top4", allowed)
        self.assertNotIn("set_budget_large", allowed)
        self.assertNotIn("verify", allowed)
        self.assertNotIn("retry", allowed)
        self.assertNotIn("terminate", allowed)

    def test_terminal_mask_blocks_retry_at_max_attempt(self):
        state = _EpisodeState(stage="terminal", attempt_index=1)
        session = Session(task_name=TaskName.OS_INTERACTION, sample_index="0")
        session.sample_status = SampleStatus.AGENT_VALIDATION_FAILED
        mask = valid_os_rl_action_mask(
            state,
            current_session=session,
            max_attempts=2,
            max_retrieval_injections=1,
            max_verifications=1,
        )
        allowed = {
            action for action, is_allowed in zip(OS_RL_ACTIONS, mask) if is_allowed
        }
        self.assertIn("terminate", allowed)
        self.assertIn("verify", allowed)
        self.assertNotIn("retry", allowed)

    def test_reward_includes_cost_retry_and_forced_terminate_penalties(self):
        reward = compute_final_reward(
            correct=True,
            total_cost_usd=0.0002,
            budget_target_usd=0.0001,
            cost_lambda=0.1,
            validation_failed=True,
            retry_count=2,
            forced_terminate=True,
        )
        self.assertAlmostEqual(reward, 1.0 - 0.2 - 0.1 - 0.06 - 0.2)

    def test_memory_hint_preserves_chat_alternation(self):
        session = Session(task_name=TaskName.OS_INTERACTION, sample_index="0")
        session.chat_history.inject(ChatHistoryItem(role=Role.USER, content="task"))
        memory = Session(task_name=TaskName.OS_INTERACTION, sample_index="memory")
        callback = _FakeRetrievalCallback(memory)
        result = append_memory_hint_to_pending_user(
            session,
            callback_dict={"retrieval": callback},
            task=object(),
            agent=_FakeAgent(),
            sample_index="0",
            replay_sample_count=1,
        )
        self.assertTrue(result["injected"])
        self.assertEqual(session.chat_history.get_value_length(), 1)
        item = session.chat_history.get_item_deep_copy(-1)
        self.assertEqual(item.role, Role.USER)
        self.assertIn("Additional retrieved memory", item.content)

    def test_verifier_without_language_model_returns_neutral(self):
        session = Session(task_name=TaskName.OS_INTERACTION, sample_index="0")
        session.chat_history.inject(ChatHistoryItem(role=Role.USER, content="task"))
        score, uncertainty, metadata = run_self_critique_verifier(
            agent=_NoLanguageModelAgent(),
            session=session,
            verifier_mode="self_critique",
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(uncertainty, 1.0)
        self.assertEqual(metadata["status"], "no_language_model")

    def test_rl_episode_smoke_with_scripted_controller(self):
        task = _TinyOSTask()
        agent = _FinishAgent()
        controller = _ScriptedController(["retrieve_top1", "reason_step", "terminate"])
        with tempfile.TemporaryDirectory() as output_dir:
            result = run_os_inference_rl_episode(
                sample_index="0",
                task=task,
                agent=agent,
                callback_handler=CallbackHandler({}),
                callback_dict={},
                session_list=[],
                cost_tracker=_FakeCostTracker(),
                controller=controller,
                output_dir=output_dir,
                runtime_state={},
                max_decisions_per_sample=4,
                max_attempts=2,
                max_retrieval_injections=1,
                max_verifications=0,
                budget_target_usd=0.0001,
                cost_lambda=0.1,
                verifier_mode="none",
            )
        self.assertEqual(result.session.sample_status, SampleStatus.COMPLETED)
        self.assertTrue(result.summary["correct"])
        self.assertIsNotNone(controller.observed_reward)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_masked_dqn_never_selects_invalid_action(self):
        config = MaskedDQNConfig(
            state_dim=len(OS_RL_FEATURE_NAMES),
            epsilon_start=0.0,
            epsilon_end=0.0,
            batch_size=1,
            seed=7,
        )
        controller = MaskedDQNInferenceController(config)
        state = [0.0 for _ in OS_RL_FEATURE_NAMES]
        valid_mask = [False for _ in OS_RL_ACTIONS]
        valid_mask[OS_RL_ACTIONS.index("terminate")] = True
        action_idx, _ = controller.select_action(state, valid_mask)
        self.assertEqual(OS_RL_ACTIONS[action_idx], "terminate")
        transition = RLEpisodeTransition(
            state=state,
            action_idx=action_idx,
            next_state=state,
            next_valid_action_mask=valid_mask,
            done=True,
        )
        update = controller.observe_episode([transition], final_reward=0.25)
        self.assertGreaterEqual(update["buffer_size"], 1)

    @unittest.skipIf(torch is None, "PyTorch is unavailable")
    def test_load_state_rejects_incompatible_config(self):
        config = MaskedDQNConfig(
            state_dim=len(OS_RL_FEATURE_NAMES),
            epsilon_start=0.0,
            epsilon_end=0.0,
            hidden_dim=16,
            seed=7,
        )
        controller = MaskedDQNInferenceController(config)
        with tempfile.TemporaryDirectory() as output_dir:
            state_path = f"{output_dir}/state.pt"
            controller.save_state(state_path)
            incompatible = MaskedDQNInferenceController(
                MaskedDQNConfig(
                    state_dim=len(OS_RL_FEATURE_NAMES),
                    epsilon_start=0.0,
                    epsilon_end=0.0,
                    hidden_dim=32,
                    seed=7,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "hidden_dim"):
                incompatible.load_state(state_path)


if __name__ == "__main__":
    unittest.main()
