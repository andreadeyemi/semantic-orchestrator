"""Natural Language Query (NLQ) module for converting queries to Pandas/SQL."""

import re
import pandas as pd
from typing import Dict, Any, List, Optional


class NLQProcessor:
    """Process natural language queries and convert to Pandas operations."""
    
    def __init__(self, dataframe: pd.DataFrame, column_info: Dict[str, Any]):
        """Initialize the NLQ processor.
        
        Args:
            dataframe: The DataFrame to query
            column_info: Metadata about DataFrame columns
        """
        self.dataframe = dataframe
        self.column_info = column_info
        self.columns = column_info["columns"]
        
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query and extract intent.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with parsed query information
        """
        query_lower = query.lower()
        
        result = {
            "original_query": query,
            "intent": "search",  # default intent
            "filters": [],
            "aggregations": [],
            "sort": None,
            "limit": None
        }
        
        # Detect aggregation intent
        if any(word in query_lower for word in ["average", "avg", "mean"]):
            result["intent"] = "aggregate"
            result["aggregations"].append("mean")
        elif any(word in query_lower for word in ["sum", "total"]):
            result["intent"] = "aggregate"
            result["aggregations"].append("sum")
        elif any(word in query_lower for word in ["count", "how many"]):
            result["intent"] = "aggregate"
            result["aggregations"].append("count")
        elif any(word in query_lower for word in ["max", "maximum", "highest"]):
            result["intent"] = "aggregate"
            result["aggregations"].append("max")
        elif any(word in query_lower for word in ["min", "minimum", "lowest"]):
            result["intent"] = "aggregate"
            result["aggregations"].append("min")
        
        # Detect sorting intent
        if "sort" in query_lower or "order" in query_lower:
            if "desc" in query_lower or "descending" in query_lower or "highest" in query_lower:
                result["sort"] = "desc"
            else:
                result["sort"] = "asc"
        
        # Detect limit
        limit_match = re.search(r'top\s+(\d+)|first\s+(\d+)|limit\s+(\d+)', query_lower)
        if limit_match:
            limit_val = next(g for g in limit_match.groups() if g is not None)
            result["limit"] = int(limit_val)
        
        # Try to extract column references
        mentioned_columns = []
        for col in self.columns:
            if col.lower() in query_lower:
                mentioned_columns.append(col)
        
        result["mentioned_columns"] = mentioned_columns
        
        return result
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a natural language query on the DataFrame.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with query results
        """
        parsed = self.parse_query(query)
        df = self.dataframe.copy()
        
        try:
            # Handle different intents
            if parsed["intent"] == "aggregate":
                result_df = self._execute_aggregation(df, parsed)
            else:
                result_df = self._execute_search(df, parsed)
            
            # Apply sorting if requested
            if parsed["sort"] and parsed["mentioned_columns"]:
                col = parsed["mentioned_columns"][0]
                ascending = parsed["sort"] == "asc"
                result_df = result_df.sort_values(by=col, ascending=ascending)
            
            # Apply limit if requested
            if parsed["limit"]:
                result_df = result_df.head(parsed["limit"])
            
            # Convert to records
            records = result_df.to_dict(orient='records')
            
            return {
                "success": True,
                "query": query,
                "parsed_intent": parsed,
                "result_count": len(records),
                "results": records,
                "pandas_code": self._generate_pandas_code(parsed)
            }
            
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "parsed_intent": parsed
            }
    
    def _execute_aggregation(self, df: pd.DataFrame, parsed: Dict[str, Any]) -> pd.DataFrame:
        """Execute aggregation query.
        
        Args:
            df: DataFrame to query
            parsed: Parsed query information
            
        Returns:
            Aggregated DataFrame
        """
        if not parsed["mentioned_columns"]:
            # Aggregate all numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                return df.head(5)
            target_cols = numeric_cols
        else:
            target_cols = parsed["mentioned_columns"]
        
        agg_func = parsed["aggregations"][0] if parsed["aggregations"] else "mean"
        
        # Perform aggregation
        result = {}
        for col in target_cols:
            if col in df.columns:
                try:
                    if agg_func == "mean":
                        result[col] = [df[col].mean()]
                    elif agg_func == "sum":
                        result[col] = [df[col].sum()]
                    elif agg_func == "count":
                        result[col] = [df[col].count()]
                    elif agg_func == "max":
                        result[col] = [df[col].max()]
                    elif agg_func == "min":
                        result[col] = [df[col].min()]
                except:
                    pass
        
        if result:
            return pd.DataFrame(result)
        return df.head(5)
    
    def _execute_search(self, df: pd.DataFrame, parsed: Dict[str, Any]) -> pd.DataFrame:
        """Execute search query.
        
        Args:
            df: DataFrame to query
            parsed: Parsed query information
            
        Returns:
            Filtered DataFrame
        """
        # For simple search, return relevant columns if mentioned
        if parsed["mentioned_columns"]:
            # Return rows with those columns
            return df[parsed["mentioned_columns"]].head(10)
        
        return df.head(10)
    
    def _generate_pandas_code(self, parsed: Dict[str, Any]) -> str:
        """Generate equivalent pandas code for the query.
        
        Args:
            parsed: Parsed query information
            
        Returns:
            String containing pandas code
        """
        code_lines = []
        
        if parsed["intent"] == "aggregate":
            agg_func = parsed["aggregations"][0] if parsed["aggregations"] else "mean"
            if parsed["mentioned_columns"]:
                cols = parsed["mentioned_columns"]
                code_lines.append(f"df[{cols}].{agg_func}()")
            else:
                code_lines.append(f"df.select_dtypes(include=['number']).{agg_func}()")
        else:
            if parsed["mentioned_columns"]:
                cols = parsed["mentioned_columns"]
                code_lines.append(f"df[{cols}]")
            else:
                code_lines.append("df")
            
            if parsed["sort"] and parsed["mentioned_columns"]:
                col = parsed["mentioned_columns"][0]
                ascending = parsed["sort"] == "asc"
                code_lines.append(f".sort_values(by='{col}', ascending={ascending})")
            
            if parsed["limit"]:
                code_lines.append(f".head({parsed['limit']})")
        
        return "".join(code_lines)
