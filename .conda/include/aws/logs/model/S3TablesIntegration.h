/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#pragma once
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/logs/CloudWatchLogs_EXPORTS.h>

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
 * <p>Contains information about the S3 Tables integration configuration for a
 * configuration template.</p><p><h3>See Also:</h3>   <a
 * href="http://docs.aws.amazon.com/goto/WebAPI/logs-2014-03-28/S3TablesIntegration">AWS
 * API Reference</a></p>
 */
class S3TablesIntegration {
 public:
  AWS_CLOUDWATCHLOGS_API S3TablesIntegration() = default;
  AWS_CLOUDWATCHLOGS_API S3TablesIntegration(Aws::Utils::Json::JsonView jsonValue);
  AWS_CLOUDWATCHLOGS_API S3TablesIntegration& operator=(Aws::Utils::Json::JsonView jsonValue);
  AWS_CLOUDWATCHLOGS_API Aws::Utils::Json::JsonValue Jsonize() const;

  ///@{
  /**
   * <p>The name of the S3 Tables datasource.</p>
   */
  inline const Aws::String& GetDatasourceName() const { return m_datasourceName; }
  inline bool DatasourceNameHasBeenSet() const { return m_datasourceNameHasBeenSet; }
  template <typename DatasourceNameT = Aws::String>
  void SetDatasourceName(DatasourceNameT&& value) {
    m_datasourceNameHasBeenSet = true;
    m_datasourceName = std::forward<DatasourceNameT>(value);
  }
  template <typename DatasourceNameT = Aws::String>
  S3TablesIntegration& WithDatasourceName(DatasourceNameT&& value) {
    SetDatasourceName(std::forward<DatasourceNameT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>The type of the S3 Tables datasource.</p>
   */
  inline const Aws::String& GetDatasourceType() const { return m_datasourceType; }
  inline bool DatasourceTypeHasBeenSet() const { return m_datasourceTypeHasBeenSet; }
  template <typename DatasourceTypeT = Aws::String>
  void SetDatasourceType(DatasourceTypeT&& value) {
    m_datasourceTypeHasBeenSet = true;
    m_datasourceType = std::forward<DatasourceTypeT>(value);
  }
  template <typename DatasourceTypeT = Aws::String>
  S3TablesIntegration& WithDatasourceType(DatasourceTypeT&& value) {
    SetDatasourceType(std::forward<DatasourceTypeT>(value));
    return *this;
  }
  ///@}
 private:
  Aws::String m_datasourceName;

  Aws::String m_datasourceType;
  bool m_datasourceNameHasBeenSet = false;
  bool m_datasourceTypeHasBeenSet = false;
};

}  // namespace Model
}  // namespace CloudWatchLogs
}  // namespace Aws
