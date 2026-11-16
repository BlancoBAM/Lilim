/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#pragma once
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/logs/CloudWatchLogs_EXPORTS.h>

namespace Aws {
namespace CloudWatchLogs {
namespace Model {
enum class DeliverySourceStatus { NOT_SET, ACTIVE, INACTIVE };

namespace DeliverySourceStatusMapper {
AWS_CLOUDWATCHLOGS_API DeliverySourceStatus GetDeliverySourceStatusForName(const Aws::String& name);

AWS_CLOUDWATCHLOGS_API Aws::String GetNameForDeliverySourceStatus(DeliverySourceStatus value);
}  // namespace DeliverySourceStatusMapper
}  // namespace Model
}  // namespace CloudWatchLogs
}  // namespace Aws
