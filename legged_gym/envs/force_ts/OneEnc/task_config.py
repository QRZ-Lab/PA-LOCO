import math
from legged_gym.envs.base.base_config import BaseConfig

# 观测维度设置
NUM_BASE_OBS = 45  # 基础观测
NUM_HISTORY = 50  # 历史观测数据长度: 50

NUM_HEIGHT_OBS = 187  # 地形观测
FORCE_HISTORY_LENGTH = 10  # 记录外力干扰时长
NUM_EXTRINSIC_OBS = 28 + 3 * FORCE_HISTORY_LENGTH # 动力学参数观测

# latent representation 变量维度设置
NUM_HEIGHT_LATENT = 16
NUM_EXTRINSIC_LATENT = 3
NUM_DISTURBANCE_LATENT = 3
NUM_LATENT = NUM_HEIGHT_LATENT + NUM_EXTRINSIC_LATENT + NUM_DISTURBANCE_LATENT


class ForceOneTSCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_base_obs = NUM_BASE_OBS
        num_height_obs = NUM_HEIGHT_OBS
        num_extrinsic_obs = NUM_EXTRINSIC_OBS

        num_history = NUM_HISTORY

        num_latent = NUM_LATENT
        num_height_latent = NUM_HEIGHT_LATENT
        num_extrinsic_latent = NUM_EXTRINSIC_LATENT
        num_disturbance_latent = NUM_DISTURBANCE_LATENT

        num_actor_obs = num_base_obs + num_latent  # actor 网络观测数
        num_observations = num_base_obs + num_height_obs + num_extrinsic_obs
        # if not None a privilege_obs_buf will be returned by step() (critic obs for asymmetric training).
        # None is returned otherwise
        num_privileged_obs = num_base_obs + num_height_obs + num_extrinsic_obs
        num_actions = 12

        env_spacing = 3.  # not used with heightfields / trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds

    class terrain:
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                             0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = -1  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        terrain_proportions = [0.1, 0.1, 0.35, 0.15, 0.25, 0., 0., 0., 0.05]
        # trimesh only:
        slope_threshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

        # new features
        plane_pre_train = True
        platform_size = 3

    class commands:
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        traj_gen_command = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            '1FR_hip_joint': -0.1,  # [rad]
            '2FL_hip_joint': 0.1,  # [rad]
            '3RR_hip_joint': -0.1,  # [rad]
            '4RL_hip_joint': 0.1,  # [rad]

            '1FR_thigh_joint': 0.8,  # [rad]
            '2FL_thigh_joint': 0.8,  # [rad]
            '3RR_thigh_joint': 1.0,  # [rad]
            '4RL_thigh_joint': 1.0,  # [rad]

            '1FR_calf_joint': -1.5,  # [rad]
            '2FL_calf_joint': -1.5,  # [rad]
            '3RR_calf_joint': -1.5,  # [rad]
            '4RL_calf_joint': -1.5,  # [rad]
        }

    class control:
        control_type = 'P'  # P: position, V: velocity, T: torques, actuator_net: network
        # PD Drive parameters:
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        default_gait_freq = 2.5
        freq_range = [2, 4]

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"  # actor name
        foot_name = "foot"  # name of the feet bodies, used to index body state and contact force tensors
        hip_name = "hip"  # name of the hip bodies
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        action_latency = False
        action_latency_timesteps = 2

        randomize_friction = True
        friction_range = [0.25, 1.5]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_com_displacement = True
        com_displacement_range_x = [-0.03, 0.03]
        com_displacement_range_y = [-0.03, 0.03]
        com_displacement_range_z = [-0.03, 0.03]

        randomize_gains = False
        p_gain_range = [0.8, 1.2]
        d_gain_range = [0.8, 1.2]

        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.
        max_push_ang_vel = 0

        continuous_push = True
        continuous_push_curriculum = True
        cont_push_start_rio = 0.6  # episode time = T; [0, rio*T) 不施加外力; [rio*T, T]: 施加外力
        cont_push_end_rio = 0.95  # episode time = T; [0, rio*T) 不施加外力; [rio*T, T]: 施加外力
        max_push_force = 1
        min_push_force = 0
        max_push_force_curriculum = 60
        push_force_step = 3
        max_push_torque = 0
        push_force_noise = 2
        push_torque_noise = 0
        force_history_length = FORCE_HISTORY_LENGTH

    class rewards:
        class scales:
            dof_pos_limits = -1.0  # -10.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            torques = -0.0002
            dof_acc = -2.5e-7
            feet_air_time = 1.0
            collision = -1.
            action_rate = -0.01

            base_height = -5
            orientation = -1

            # action_smoothness = -0.001
            feet_down_hip_global = 0.3

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95  # 0.9; percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.28
        max_contact_force = 100.  # forces above this value are penalized

        feet_down_hip_sigma = 0.02  # reward = sum{feet}( exp( - error_x^2 + error_y^2 / sigma^2 )) / 4
        feet_hip_offset = [0.0, -0.08, 0.0, 0.08, -0.0, -0.08, -0.0, 0.08]  # FR_xy / FL_xy / RR_xy / RL_xy

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class ForceOneTSCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'TeacherRunner'
    student_runner_class_name = 'StudentRunner'
    residual_runner_class_name = 'ResidualRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_activation = None
        # only for 'ActorCriticRecurrent':
        rnn_type = 'gru'  # 'gru', 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCriticMlpEncoder'
        student_policy_class_name = 'ActorCriticMlpEncoder'
        residual_policy_class_name = 'ActorCriticMlpEncoderResidual'
        algorithm_class_name = 'PPO'
        # student_algorithm_class_name = 'PPO'
        residual_algorithm_class_name = 'PPOResidual'
        num_steps_per_env = 24  # per iteration
        max_iterations = 6000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = 'force_one_t'
        run_name = ''

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

        # student runner 设置
        fine_tune = True  # True 则微调基础策略网络 (actor network); False 则固定 student 网络中基础策略网络的参数; 计算 loss 时也会发生相应的更改
        learning_rate = 1e-4  # student 网络的 learning rate; 学习率衰退: lr = lr0 * a.pow(updates/b)
        exp_decay_a = 0.995
        exp_decay_b = 10

    class encoder:
        is_teacher = True
        encoder_output_tanh = False

        # %%%%%%% Teacher Encoder Setting  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # (only one) mlp encoder for teacher
        mlp_input_dim_e_t = NUM_HEIGHT_OBS + NUM_EXTRINSIC_OBS
        mlp_hidden_dims_e_t = [256, 128]
        mlp_output_dim_e_t = NUM_LATENT

        # %%%%%%% Student Encoder Setting  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # mlp encoder for student
        mlp_input_dim_e_s = NUM_BASE_OBS * NUM_HISTORY
        mlp_hidden_dims_e_s = [1024, 512, 256]  # [1024, 512, 256, 128]
        mlp_output_dim_e_s = NUM_LATENT

        # tcn encoder for student
        tcn_channels = [32, 32, 32]
        tcn_kernel_size = 3
        tcn_dropout = 0.2
