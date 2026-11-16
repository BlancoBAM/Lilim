/**
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0.
 */

#pragma once
#include <aws/core/utils/memory/stl/AWSString.h>
#include <aws/core/utils/memory/stl/AWSVector.h>
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
 * <p>A tag filter that specifies a tag key and optional tag values for filtering
 * log groups by tags.</p><p><h3>See Also:</h3>   <a
 * href="http://docs.aws.amazon.com/goto/WebAPI/logs-2014-03-28/TagFilter">AWS API
 * Reference</a></p>
 */
class TagFilter {
 public:
  AWS_CLOUDWATCHLOGS_API TagFilter() = default;
  AWS_CLOUDWATCHLOGS_API TagFilter(Aws::Utils::Json::JsonView jsonValue);
  AWS_CLOUDWATCHLOGS_API TagFilter& operator=(Aws::Utils::Json::JsonView jsonValue);
  AWS_CLOUDWATCHLOGS_API Aws::Utils::Json::JsonValue Jsonize() const;

  ///@{
  /**
   * <p>The tag key to filter on.</p>
   */
  inline const Aws::String& GetKey() const { return m_key; }
  inline bool KeyHasBeenSet() const { return m_keyHasBeenSet; }
  template <typename KeyT = Aws::String>
  void SetKey(KeyT&& value) {
    m_keyHasBeenSet = true;
    m_key = std::forward<KeyT>(value);
  }
  template <typename KeyT = Aws::String>
  TagFilter& WithKey(KeyT&& value) {
    SetKey(std::forward<KeyT>(value));
    return *this;
  }
  ///@}

  ///@{
  /**
   * <p>An optional list of tag values to filter on.</p> <ul> <li> <p>If you specify
   * a filter that contains more than one value for a key, the response returns log
   * groups that match any of the specified values for that key.</p> </li> <li> <p>If
   * you don't specify values, the response returns all log groups that are tagged
   * with that key, with any or no value.</p> </li> <li> <p>Use <code>*</code> for
   * wildcard matching. For example, <code>prod*</code> matches values that start
   * with <code>prod</code>.</p> </li> <li> <p>Use <code>!</code> as a prefix for
   * negation. For example, <code>!prod</code> matches values that are not
   * <code>prod</code>.</p> </li> <li> <p>Exact matching and negation are
   * case-sensitive. Wildcard matching is case-insensitive.</p> </li> </ul>
   */
  inline const Aws::Vector<Aws::String>& GetValues() const { return m_values; }
  inline bool ValuesHasBeenSet() const { return m_valuesHasBeenSet; }
  template <typename ValuesT = Aws::Vector<Aws::String>>
  void SetValues(ValuesT&& value) {
    m_valuesHasBeenSet = true;
    m_values = std::forward<ValuesT>(value);
  }
  template <typename ValuesT = Aws::Vector<Aws::String>>
  TagFilter& WithValues(ValuesT&& value) {
    SetValues(std::forward<ValuesT>(value));
    return *this;
  }
  template <typename ValuesT = Aws::String>
  TagFilter& AddValues(ValuesT&& value) {
    m_valuesHasBeenSet = true;
    m_values.emplace_back(std::forward<ValuesT>(value));
    return *this;
  }
  ///@}
 private:
  Aws::String m_key;

  Aws::Vector<Aws::String> m_values;
  bool m_keyHasBeenSet = false;
  bool m_valuesHasBeenSet = false;
};

}  // namespace Model
}  // namespace CloudWatchLogs
}  // namespace Aws
