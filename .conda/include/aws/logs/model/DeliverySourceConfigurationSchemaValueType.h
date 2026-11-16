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
enum class DeliverySourceConfigurationSchemaValueType { NOT_SET, string, boolean, int_, double_, long_ };

namespace DeliverySourceConfigurationSchemaValueTypeMapper {
AWS_CLOUDWATCHLOGS_API DeliverySourceConfigurationSchemaValueType
GetDeliverySourceConfigurationSchemaValueTypeForName(const Aws::String& name);

AWS_CLOUDWATCHLOGS_API Aws::String GetNameForDeliverySourceConfigurationSchemaValueType(DeliverySourceConfigurationSchemaValueType value);
}  // namespace DeliverySourceConfigurationSchemaValueTypeMapper
}  // namespace Model
}  // namespace CloudWatchLogs
}  // namespace Aws
