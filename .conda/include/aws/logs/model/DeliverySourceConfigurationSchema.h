/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#pragma once
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
#include <aws/logs/CloudWatchLogs_EXPORTS.h>
#include <aws/logs/model/DeliverySourceConfigurationSchemaValueType.h>

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
 * <p>A structure that describes a single configuration for a log type, including
 * its name, value type, default value, and the range of supported
 * values.</p><p><h3>See Also:</h3>   <a
 * href="http://docs.aws.amazon.com/goto/WebAPI/logs-2014-03-28/DeliverySourceConfigurationSchema">AWS
 * API Reference</a></p>
 */
class DeliverySourceConfigurationSchema {
 public:
  AWS_CLOUDWATCHLOGS_API DeliverySourceConfigurationSchema() = default;
  AWS_CLOUDWATCHLOGS_API DeliverySourceConfigurationSchema(Aws::Utils::Json::JsonView jsonValue);
  AWS_CLOUDWATCHLOGS_API DeliverySourceConfigurationSchema& operator=(Aws::Utils::Json::JsonView jsonValue);
  AWS_CLOUDWATCHLOGS_API Aws::Utils::Json::JsonValue Jsonize() const;

  ///@{
  /**
   * <p>The name of the configuration.</p>
   */
  inline const Aws::String& GetKeyName() const { return m_keyName; }
  inline bool KeyNameHasBeenSet() const { return m_keyNameHasBeenSet; }
  template <typename KeyNameT = Aws::String>
  void SetKeyName(KeyNameT&& value) {
    m_keyNameHasBeenSet = true;
    m_keyName = std::forward<KeyNameT>(value);
  }
  template <typename KeyNameT = Aws::String>
  DeliverySourceConfigurationSchema& WithKeyName(KeyNameT&& value) {
    SetKeyName(std::forward<KeyNameT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The data type of the configuration value. Valid values are
   * <code>string</code>, <code>boolean</code>, <code>int</code>,
   * <code>double</code>, and <code>long</code>.</p>
   */
  inline DeliverySourceConfigurationSchemaValueType GetValueType() const { return m_valueType; }
  inline bool ValueTypeHasBeenSet() const { return m_valueTypeHasBeenSet; }
  inline void SetValueType(DeliverySourceConfigurationSchemaValueType value) {
    m_valueTypeHasBeenSet = true;
    m_valueType = value;
  }
  inline DeliverySourceConfigurationSchema& WithValueType(DeliverySourceConfigurationSchemaValueType value) {
    SetValueType(value);
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The default value of the configuration that is used when a value is not
   * specified in a <a
   * href="https://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_PutDeliverySource.html">PutDeliverySource</a>
   * request.</p>
   */
  inline const Aws::String& GetDefaultValue() const { return m_defaultValue; }
  inline bool DefaultValueHasBeenSet() const { return m_defaultValueHasBeenSet; }
  template <typename DefaultValueT = Aws::String>
  void SetDefaultValue(DefaultValueT&& value) {
    m_defaultValueHasBeenSet = true;
    m_defaultValue = std::forward<DefaultValueT>(value);
  }
  template <typename DefaultValueT = Aws::String>
  DeliverySourceConfigurationSchema& WithDefaultValue(DefaultValueT&& value) {
    SetDefaultValue(std::forward<DefaultValueT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The list of allowed values for the configuration. Empty for free-form
   * configuration.</p>
   */
  inline const Aws::Vector<Aws::String>& GetSupportedValues() const { return m_supportedValues; }
  inline bool SupportedValuesHasBeenSet() const { return m_supportedValuesHasBeenSet; }
  template <typename SupportedValuesT = Aws::Vector<Aws::String>>
  void SetSupportedValues(SupportedValuesT&& value) {
    m_supportedValuesHasBeenSet = true;
    m_supportedValues = std::forward<SupportedValuesT>(value);
  }
  template <typename SupportedValuesT = Aws::Vector<Aws::String>>
  DeliverySourceConfigurationSchema& WithSupportedValues(SupportedValuesT&& value) {
    SetSupportedValues(std::forward<SupportedValuesT>(value));
    return *this;
  }
  template <typename SupportedValuesT = Aws::String>
  DeliverySourceConfigurationSchema& AddSupportedValues(SupportedValuesT&& value) {
    m_supportedValuesHasBeenSet = true;
    m_supportedValues.emplace_back(std::forward<SupportedValuesT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The minimum numeric value allowed for the configuration. This applies only
   * when the <code>valueType</code> is a numeric type.</p>
   */
  inline double GetMinValue() const { return m_minValue; }
  inline bool MinValueHasBeenSet() const { return m_minValueHasBeenSet; }
  inline void SetMinValue(double value) {
    m_minValueHasBeenSet = true;
    m_minValue = value;
  }
  inline DeliverySourceConfigurationSchema& WithMinValue(double value) {
    SetMinValue(value);
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The maximum numeric value allowed for the configuration. This applies only
   * when the <code>valueType</code> is a numeric type.</p>
   */
  inline double GetMaxValue() const { return m_maxValue; }
  inline bool MaxValueHasBeenSet() const { return m_maxValueHasBeenSet; }
  inline void SetMaxValue(double value) {
    m_maxValueHasBeenSet = true;
    m_maxValue = value;
  }
  inline DeliverySourceConfigurationSchema& WithMaxValue(double value) {
    SetMaxValue(value);
    return *this;
  }
  ///@}
 private:
  Aws::String m_keyName;

  DeliverySourceConfigurationSchemaValueType m_valueType{DeliverySourceConfigurationSchemaValueType::NOT_SET};

  Aws::String m_defaultValue;

  Aws::Vector<Aws::String> m_supportedValues;

  double m_minValue{0.0};

  double m_maxValue{0.0};
  bool m_keyNameHasBeenSet = false;
  bool m_valueTypeHasBeenSet = false;
  bool m_defaultValueHasBeenSet = false;
  bool m_supportedValuesHasBeenSet = false;
  bool m_minValueHasBeenSet = false;
  bool m_maxValueHasBeenSet = false;
};

}  // namespace Model
}  // namespace CloudWatchLogs
}  // namespace Aws
