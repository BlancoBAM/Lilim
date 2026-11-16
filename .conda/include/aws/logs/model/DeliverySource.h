/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#pragma once
#include <aws/core/utils/memory/stl/AWSMap.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/logs/CloudWatchLogs_EXPORTS.h>
#include <aws/logs/model/DeliverySourceStatus.h>
#include <aws/logs/model/DeliverySourceStatusReason.h>

#include <utility>

namespace Aws {
namespace Utils {
namespace Json {
class JsonValue;
class JsonView;
}  // namespace Json
}  // namespace Utils
namespace CloudWatchLogs {
namespace Model {

/**
 * <p>This structure contains information about one <i>delivery source</i> in your
 * account. A delivery source is an Amazon Web Services resource that sends logs to
 * an Amazon Web Services destination. The destination can be CloudWatch Logs,
 * Amazon S3, or Firehose.</p> <p>Only some Amazon Web Services services support
 * being configured as a delivery source. These services are listed as <b>Supported
 * [V2 Permissions]</b> in the table at <a
 * href="https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AWS-logs-and-resource-policy.html">Enabling
 * logging from Amazon Web Services services.</a> </p> <p>To configure logs
 * delivery between a supported Amazon Web Services service and a destination, you
 * must do the following:</p> <ul> <li> <p>Create a delivery source, which is a
 * logical object that represents the resource that is actually sending the logs.
 * For more information, see <a
 * href="https://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_PutDeliverySource.html">PutDeliverySource</a>.</p>
 * </li> <li> <p>Create a <i>delivery destination</i>, which is a logical object
 * that represents the actual delivery destination. For more information, see <a
 * href="https://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_PutDeliveryDestination.html">PutDeliveryDestination</a>.</p>
 * </li> <li> <p>If you are delivering logs cross-account, you must use <a
 * href="https://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_PutDeliveryDestinationPolicy.html">PutDeliveryDestinationPolicy</a>
 * in the destination account to assign an IAM policy to the destination. This
 * policy allows delivery to that destination. </p> </li> <li> <p>Create a
 * <i>delivery</i> by pairing exactly one delivery source and one delivery
 * destination. For more information, see <a
 * href="https://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_CreateDelivery.html">CreateDelivery</a>.</p>
 * </li> </ul> <p>You can configure a single delivery source to send logs to
 * multiple destinations by creating multiple deliveries. You can also create
 * multiple deliveries to configure multiple delivery sources to send logs to the
 * same delivery destination.</p><p><h3>See Also:</h3>   <a
 * href="http://docs.aws.amazon.com/goto/WebAPI/logs-2014-03-28/DeliverySource">AWS
 * API Reference</a></p>
 */
class DeliverySource {
 public:
  AWS_CLOUDWATCHLOGS_API DeliverySource() = default;
  AWS_CLOUDWATCHLOGS_API DeliverySource(Aws::Utils::Json::JsonView jsonValue);
  AWS_CLOUDWATCHLOGS_API DeliverySource& operator=(Aws::Utils::Json::JsonView jsonValue);
  AWS_CLOUDWATCHLOGS_API Aws::Utils::Json::JsonValue Jsonize() const;

  ///@{
  /**
   * <p>The unique name of the delivery source.</p>
   */
  inline const Aws::String& GetName() const { return m_name; }
  inline bool NameHasBeenSet() const { return m_nameHasBeenSet; }
  template <typename NameT = Aws::String>
  void SetName(NameT&& value) {
    m_nameHasBeenSet = true;
    m_name = std::forward<NameT>(value);
  }
  template <typename NameT = Aws::String>
  DeliverySource& WithName(NameT&& value) {
    SetName(std::forward<NameT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The Amazon Resource Name (ARN) that uniquely identifies this delivery
   * source.</p>
   */
  inline const Aws::String& GetArn() const { return m_arn; }
  inline bool ArnHasBeenSet() const { return m_arnHasBeenSet; }
  template <typename ArnT = Aws::String>
  void SetArn(ArnT&& value) {
    m_arnHasBeenSet = true;
    m_arn = std::forward<ArnT>(value);
  }
  template <typename ArnT = Aws::String>
  DeliverySource& WithArn(ArnT&& value) {
    SetArn(std::forward<ArnT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>This array contains the ARN of the Amazon Web Services resource that sends
   * logs and is represented by this delivery source. Currently, only one ARN can be
   * in the array.</p>
   */
  inline const Aws::Vector<Aws::String>& GetResourceArns() const { return m_resourceArns; }
  inline bool ResourceArnsHasBeenSet() const { return m_resourceArnsHasBeenSet; }
  template <typename ResourceArnsT = Aws::Vector<Aws::String>>
  void SetResourceArns(ResourceArnsT&& value) {
    m_resourceArnsHasBeenSet = true;
    m_resourceArns = std::forward<ResourceArnsT>(value);
  }
  template <typename ResourceArnsT = Aws::Vector<Aws::String>>
  DeliverySource& WithResourceArns(ResourceArnsT&& value) {
    SetResourceArns(std::forward<ResourceArnsT>(value));
    return *this;
  }
  template <typename ResourceArnsT = Aws::String>
  DeliverySource& AddResourceArns(ResourceArnsT&& value) {
    m_resourceArnsHasBeenSet = true;
    m_resourceArns.emplace_back(std::forward<ResourceArnsT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The Amazon Web Services service that is sending logs.</p>
   */
  inline const Aws::String& GetService() const { return m_service; }
  inline bool ServiceHasBeenSet() const { return m_serviceHasBeenSet; }
  template <typename ServiceT = Aws::String>
  void SetService(ServiceT&& value) {
    m_serviceHasBeenSet = true;
    m_service = std::forward<ServiceT>(value);
  }
  template <typename ServiceT = Aws::String>
  DeliverySource& WithService(ServiceT&& value) {
    SetService(std::forward<ServiceT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The type of log that the source is sending. For valid values for this
   * parameter, see the documentation for the source service.</p>
   */
  inline const Aws::String& GetLogType() const { return m_logType; }
  inline bool LogTypeHasBeenSet() const { return m_logTypeHasBeenSet; }
  template <typename LogTypeT = Aws::String>
  void SetLogType(LogTypeT&& value) {
    m_logTypeHasBeenSet = true;
    m_logType = std::forward<LogTypeT>(value);
  }
  template <typename LogTypeT = Aws::String>
  DeliverySource& WithLogType(LogTypeT&& value) {
    SetLogType(std::forward<LogTypeT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The tags that have been assigned to this delivery source.</p>
   */
  inline const Aws::Map<Aws::String, Aws::String>& GetTags() const { return m_tags; }
  inline bool TagsHasBeenSet() const { return m_tagsHasBeenSet; }
  template <typename TagsT = Aws::Map<Aws::String, Aws::String>>
  void SetTags(TagsT&& value) {
    m_tagsHasBeenSet = true;
    m_tags = std::forward<TagsT>(value);
  }
  template <typename TagsT = Aws::Map<Aws::String, Aws::String>>
  DeliverySource& WithTags(TagsT&& value) {
    SetTags(std::forward<TagsT>(value));
    return *this;
  }
  template <typename TagsKeyT = Aws::String, typename TagsValueT = Aws::String>
  DeliverySource& AddTags(TagsKeyT&& key, TagsValueT&& value) {
    m_tagsHasBeenSet = true;
    m_tags.emplace(std::forward<TagsKeyT>(key), std::forward<TagsValueT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The map of key-value pairs that configure the delivery source.</p>
   */
  inline const Aws::Map<Aws::String, Aws::String>& GetDeliverySourceConfiguration() const { return m_deliverySourceConfiguration; }
  inline bool DeliverySourceConfigurationHasBeenSet() const { return m_deliverySourceConfigurationHasBeenSet; }
  template <typename DeliverySourceConfigurationT = Aws::Map<Aws::String, Aws::String>>
  void SetDeliverySourceConfiguration(DeliverySourceConfigurationT&& value) {
    m_deliverySourceConfigurationHasBeenSet = true;
    m_deliverySourceConfiguration = std::forward<DeliverySourceConfigurationT>(value);
  }
  template <typename DeliverySourceConfigurationT = Aws::Map<Aws::String, Aws::String>>
  DeliverySource& WithDeliverySourceConfiguration(DeliverySourceConfigurationT&& value) {
    SetDeliverySourceConfiguration(std::forward<DeliverySourceConfigurationT>(value));
    return *this;
  }
  template <typename DeliverySourceConfigurationKeyT = Aws::String, typename DeliverySourceConfigurationValueT = Aws::String>
  DeliverySource& AddDeliverySourceConfiguration(DeliverySourceConfigurationKeyT&& key, DeliverySourceConfigurationValueT&& value) {
    m_deliverySourceConfigurationHasBeenSet = true;
    m_deliverySourceConfiguration.emplace(std::forward<DeliverySourceConfigurationKeyT>(key),
                                          std::forward<DeliverySourceConfigurationValueT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The status of the delivery source. A delivery source can have the status
   * <code>ACTIVE</code> or <code>INACTIVE</code>. Note: This value is defined for
   * selective log types.</p>
   */
  inline DeliverySourceStatus GetStatus() const { return m_status; }
  inline bool StatusHasBeenSet() const { return m_statusHasBeenSet; }
  inline void SetStatus(DeliverySourceStatus value) {
    m_statusHasBeenSet = true;
    m_status = value;
  }
  inline DeliverySource& WithStatus(DeliverySourceStatus value) {
    SetStatus(value);
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The reason for the status of the delivery source. A status reason of
   * <code>RESOURCE_DELETED</code> indicates that the resource associated with the
   * delivery source has been deleted. Note: This value is defined for selective log
   * types.</p>
   */
  inline DeliverySourceStatusReason GetStatusReason() const { return m_statusReason; }
  inline bool StatusReasonHasBeenSet() const { return m_statusReasonHasBeenSet; }
  inline void SetStatusReason(DeliverySourceStatusReason value) {
    m_statusReasonHasBeenSet = true;
    m_statusReason = value;
  }
  inline DeliverySource& WithStatusReason(DeliverySourceStatusReason value) {
    SetStatusReason(value);
    return *this;
  }
  ///@}
 private:
  Aws::String m_name;

  Aws::String m_arn;

  Aws::Vector<Aws::String> m_resourceArns;

  Aws::String m_service;

  Aws::String m_logType;

  Aws::Map<Aws::String, Aws::String> m_tags;

  Aws::Map<Aws::String, Aws::String> m_deliverySourceConfiguration;

  DeliverySourceStatus m_status{DeliverySourceStatus::NOT_SET};

  DeliverySourceStatusReason m_statusReason{DeliverySourceStatusReason::NOT_SET};
  bool m_nameHasBeenSet = false;
  bool m_arnHasBeenSet = false;
  bool m_resourceArnsHasBeenSet = false;
  bool m_serviceHasBeenSet = false;
  bool m_logTypeHasBeenSet = false;
  bool m_tagsHasBeenSet = false;
  bool m_deliverySourceConfigurationHasBeenSet = false;
  bool m_statusHasBeenSet = false;
  bool m_statusReasonHasBeenSet = false;
};

}  // namespace Model
}  // namespace CloudWatchLogs
}  // namespace Aws
