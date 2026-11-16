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
enum class DeliverySourceStatusReason { NOT_SET, RESOURCE_DELETED };

namespace DeliverySourceStatusReasonMapper {
AWS_CLOUDWATCHLOGS_API DeliverySourceStatusReason GetDeliverySourceStatusReasonForName(const Aws::String& name);

AWS_CLOUDWATCHLOGS_API Aws::String GetNameForDeliverySourceStatusReason(DeliverySourceStatusReason value);
}  // namespace DeliverySourceStatusReasonMapper
}  // namespace Model
}  // namespace CloudWatchLogs
}  // namespace Aws
