-- =============================================================================
-- 宏观周期分析平台 - MySQL 表结构
-- 依据：毕业设计文档 033121234.pdf（表 4-1 ~ 表 4-7）
-- 字符集：utf8mb4（兼容中文与 emoji）
-- =============================================================================

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- -----------------------------------------------------------------------------
-- 表 4-1 用户信息表
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS `indicator_weights`;
DROP TABLE IF EXISTS `analysis_results`;
DROP TABLE IF EXISTS `system_logs`;
DROP TABLE IF EXISTS `analysis_tasks`;
DROP TABLE IF EXISTS `datasets`;
DROP TABLE IF EXISTS `model_configs`;
DROP TABLE IF EXISTS `users`;

CREATE TABLE `users` (
  `id`            INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '用户 ID',
  `username`      VARCHAR(255) NOT NULL COMMENT '用户名',
  `password`      VARCHAR(255) NOT NULL COMMENT '用户密码（建议存哈希）',
  `is_admin`      TINYINT(1)   NOT NULL DEFAULT 0 COMMENT '是否为管理员（0=普通用户，1=管理员）',
  `analysis_count` INT         NOT NULL DEFAULT 0 COMMENT '分析次数',
  `create_time`   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '注册时间',
  `update_time`   DATETIME     NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '信息修改时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_users_username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户信息表';

-- -----------------------------------------------------------------------------
-- 表 4-2 数据集表
-- -----------------------------------------------------------------------------
CREATE TABLE `datasets` (
  `id`         INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '数据集 ID',
  `user_id`    INT UNSIGNED NOT NULL COMMENT '所属用户 ID',
  `name`       VARCHAR(255) NOT NULL COMMENT '数据集名称',
  `row_count`  INT UNSIGNED NOT NULL DEFAULT 0 COMMENT '数据行数',
  `create_time` DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `source`     VARCHAR(50)  NULL DEFAULT NULL COMMENT '数据来源（upload/crawl 等）',
  PRIMARY KEY (`id`),
  KEY `idx_datasets_user_id` (`user_id`),
  CONSTRAINT `fk_datasets_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
    ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='数据集表';

-- -----------------------------------------------------------------------------
-- 表 4-3 分析任务表
-- -----------------------------------------------------------------------------
CREATE TABLE `analysis_tasks` (
  `id`             INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '分析任务 ID（run_id）',
  `user_id`        INT UNSIGNED NOT NULL COMMENT '所属用户 ID',
  `dataset_id`     INT UNSIGNED NOT NULL COMMENT '关联数据集 ID',
  `predict_model`  VARCHAR(50)  NOT NULL COMMENT '预测模型类型（如 hw/lstm）',
  `fusion_method`  VARCHAR(50)  NOT NULL COMMENT '指标融合方式（如 pca/dfm/entropy/equal）',
  `predict_months` INT UNSIGNED NOT NULL DEFAULT 3 COMMENT '预测月数',
  `create_time`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '分析创建时间',
  `status`         VARCHAR(20)  NOT NULL DEFAULT 'running' COMMENT '分析状态（completed/running/failed 等）',
  PRIMARY KEY (`id`),
  KEY `idx_analysis_tasks_user_id` (`user_id`),
  KEY `idx_analysis_tasks_dataset_id` (`dataset_id`),
  CONSTRAINT `fk_analysis_tasks_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
    ON DELETE RESTRICT ON UPDATE CASCADE,
  CONSTRAINT `fk_analysis_tasks_dataset` FOREIGN KEY (`dataset_id`) REFERENCES `datasets` (`id`)
    ON DELETE RESTRICT ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='分析任务表';

-- -----------------------------------------------------------------------------
-- 表 4-4 模型配置表
-- -----------------------------------------------------------------------------
CREATE TABLE `model_configs` (
  `id`          INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '模型 ID',
  `type`        VARCHAR(50)  NOT NULL COMMENT '模型类型（fusion/forecast 等）',
  `name`        VARCHAR(100) NOT NULL COMMENT '模型名称',
  `is_enabled`  TINYINT(1)   NOT NULL DEFAULT 1 COMMENT '是否启用（0=禁用，1=启用）',
  `params`      TEXT         NULL COMMENT '模型参数配置（JSON 字符串）',
  `create_time` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` DATETIME     NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_model_configs_type` (`type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='模型配置表';

-- -----------------------------------------------------------------------------
-- 表 4-5 系统日志表
-- -----------------------------------------------------------------------------
CREATE TABLE `system_logs` (
  `id`           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '日志 ID',
  `user_id`      INT UNSIGNED NULL DEFAULT NULL COMMENT '操作用户 ID（匿名接口可为空）',
  `username`     VARCHAR(255) NOT NULL DEFAULT '' COMMENT '操作用户名',
  `operation`    VARCHAR(100) NOT NULL COMMENT '操作内容',
  `detail`       TEXT         NULL COMMENT '操作详情',
  `status_code`  INT          NOT NULL DEFAULT 200 COMMENT '响应状态码',
  `operate_time` DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '操作时间',
  `ip`           VARCHAR(50)  NULL DEFAULT NULL COMMENT '操作 IP 地址',
  PRIMARY KEY (`id`),
  KEY `idx_system_logs_user_id` (`user_id`),
  KEY `idx_system_logs_operate_time` (`operate_time`),
  CONSTRAINT `fk_system_logs_user` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`)
    ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统日志表';

-- -----------------------------------------------------------------------------
-- 表 4-6 分析结果表
-- -----------------------------------------------------------------------------
CREATE TABLE `analysis_results` (
  `id`               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '结果记录 ID',
  `run_id`           INT UNSIGNED NOT NULL COMMENT '关联分析任务 ID',
  `date`             DATE           NOT NULL COMMENT '数据日期',
  `composite_index`  DECIMAL(10,4)  NOT NULL COMMENT '综合指数',
  `stage`            VARCHAR(20)    NOT NULL COMMENT '周期阶段（扩张/收缩/低谷/滞涨 等）',
  `is_history`       TINYINT(1)     NOT NULL COMMENT '是否为历史数据（0=预测，1=历史）',
  `create_time`      DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '结果生成时间',
  PRIMARY KEY (`id`),
  KEY `idx_analysis_results_run_id` (`run_id`),
  KEY `idx_analysis_results_run_date` (`run_id`, `date`),
  CONSTRAINT `fk_analysis_results_task` FOREIGN KEY (`run_id`) REFERENCES `analysis_tasks` (`id`)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='分析结果表';

-- -----------------------------------------------------------------------------
-- 表 4-7 指标权重表
-- -----------------------------------------------------------------------------
CREATE TABLE `indicator_weights` (
  `id`             BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '权重记录 ID',
  `run_id`         INT UNSIGNED NOT NULL COMMENT '关联分析任务 ID',
  `indicator_name` VARCHAR(100) NOT NULL COMMENT '指标名称',
  `weight`         DECIMAL(10,4) NOT NULL COMMENT '指标权重值',
  `create_time`    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '权重计算时间',
  PRIMARY KEY (`id`),
  KEY `idx_indicator_weights_run_id` (`run_id`),
  CONSTRAINT `fk_indicator_weights_task` FOREIGN KEY (`run_id`) REFERENCES `analysis_tasks` (`id`)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='指标权重表';

SET FOREIGN_KEY_CHECKS = 1;
