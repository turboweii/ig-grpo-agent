#!/usr/bin/env python3
"""
生成 τ-bench Retail 工具配置
基于 TauBench 的工具定义生成 YAML 配置
"""
import json
from pathlib import Path


# Retail 域工具定义 (基于 tau-bench)
RETAIL_TOOLS = [
    {
        "name": "Calculate",
        "description": "Calculate the result of a math operation.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to calculate, e.g., '5 + 3', '10 * 2'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "CancelPendingOrder",
        "description": "Cancel a pending order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The ID of the order to cancel"
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "ExchangeDeliveredOrderItems",
        "description": "Exchange items from a delivered order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The ID of the delivered order"
                },
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of item IDs to exchange"
                },
                "replacement_item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of replacement item IDs"
                }
            },
            "required": ["order_id", "item_ids", "replacement_item_ids"]
        }
    },
    {
        "name": "FindUserIdByEmail",
        "description": "Find user ID by email address.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The email address to search for"
                }
            },
            "required": ["email"]
        }
    },
    {
        "name": "FindUserIdByNameZip",
        "description": "Find user ID by name and zip code.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The user's full name"
                },
                "zip_code": {
                    "type": "string",
                    "description": "The user's zip code"
                }
            },
            "required": ["name", "zip_code"]
        }
    },
    {
        "name": "GetOrderDetails",
        "description": "Get details of a specific order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The ID of the order"
                }
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "GetProductDetails",
        "description": "Get details of a specific product.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The ID of the product"
                }
            },
            "required": ["product_id"]
        }
    },
    {
        "name": "GetUserDetails",
        "description": "Get details of a specific user.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The ID of the user"
                }
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "ListAllProductTypes",
        "description": "List all available product types.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "ModifyPendingOrderAddress",
        "description": "Modify the shipping address of a pending order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The ID of the pending order"
                },
                "new_address": {
                    "type": "string",
                    "description": "The new shipping address"
                },
                "new_city": {
                    "type": "string",
                    "description": "The new city"
                },
                "new_state": {
                    "type": "string",
                    "description": "The new state"
                },
                "new_zip_code": {
                    "type": "string",
                    "description": "The new zip code"
                }
            },
            "required": ["order_id", "new_address", "new_city", "new_state", "new_zip_code"]
        }
    },
    {
        "name": "ModifyPendingOrderItems",
        "description": "Modify items in a pending order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The ID of the pending order"
                },
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of item IDs to modify"
                },
                "quantities": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "New quantities for the items"
                }
            },
            "required": ["order_id", "item_ids", "quantities"]
        }
    },
    {
        "name": "ModifyPendingOrderPayment",
        "description": "Modify the payment method of a pending order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The ID of the pending order"
                },
                "payment_method": {
                    "type": "string",
                    "description": "The new payment method (e.g., 'credit_card', 'paypal')"
                }
            },
            "required": ["order_id", "payment_method"]
        }
    },
    {
        "name": "ModifyUserAddress",
        "description": "Modify a user's address.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The ID of the user"
                },
                "new_address": {
                    "type": "string",
                    "description": "The new address"
                },
                "new_city": {
                    "type": "string",
                    "description": "The new city"
                },
                "new_state": {
                    "type": "string",
                    "description": "The new state"
                },
                "new_zip_code": {
                    "type": "string",
                    "description": "The new zip code"
                }
            },
            "required": ["user_id", "new_address", "new_city", "new_state", "new_zip_code"]
        }
    },
    {
        "name": "ReturnDeliveredOrderItems",
        "description": "Return items from a delivered order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The ID of the delivered order"
                },
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of item IDs to return"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for return"
                }
            },
            "required": ["order_id", "item_ids"]
        }
    },
    {
        "name": "Think",
        "description": "Think about the next action. This tool allows you to reason before taking action.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your thought process"
                }
            },
            "required": ["thought"]
        }
    },
    {
        "name": "TransferToHumanAgents",
        "description": "Transfer the conversation to a human agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for transfer"
                }
            },
            "required": ["reason"]
        }
    },
]


def generate_tool_config():
    """生成工具配置 YAML"""
    output = []

    for tool in RETAIL_TOOLS:
        tool_entry = {
            "class_name": f"src.envs.tau_bench_tools.TauBench_{tool['name']}_Tool",
            "config": {"type": "native"},
            "tool_schema": {
                "type": "function",
                "function": {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": tool['parameters']
                }
            }
        }
        output.append(tool_entry)

    return output


def write_yaml(data, file_path):
    """简单的 YAML 写入器"""
    with open(file_path, 'w') as f:
        f.write("tools:\n")
        for tool in data:
            f.write(f"- class_name: {tool['class_name']}\n")
            f.write("  config:\n")
            f.write(f"    type: {tool['config']['type']}\n")
            f.write("  tool_schema:\n")
            f.write(f"    type: {tool['tool_schema']['type']}\n")
            f.write("    function:\n")
            f.write(f"      name: {tool['tool_schema']['function']['name']}\n")
            f.write(f"      description: {tool['tool_schema']['function']['description']}\n")
            f.write("      parameters:\n")

            params = tool['tool_schema']['function']['parameters']
            f.write(f"        type: {params['type']}\n")

            if 'properties' in params:
                f.write("        properties:\n")
                for prop_name, prop_info in params['properties'].items():
                    f.write(f"          {prop_name}:\n")
                    f.write(f"            type: {prop_info['type']}\n")
                    if 'description' in prop_info:
                        f.write(f"            description: {prop_info['description']}\n")
                    if 'enum' in prop_info:
                        f.write(f"            enum:\n")
                        for e in prop_info['enum']:
                            f.write(f"            - {e}\n")
                    if 'items' in prop_info and 'type' in prop_info['items']:
                        f.write(f"            items:\n")
                        f.write(f"              type: {prop_info['items']['type']}\n")

            if 'required' in params:
                f.write(f"        required: {params['required']}\n")


def main():
    output_path = Path("configs/tool_config/tau_bench_retail_tools.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tools = generate_tool_config()

    # 写入 YAML
    write_yaml(tools, output_path)

    print(f"Retail 工具配置已生成: {output_path}")
    print(f"共 {len(RETAIL_TOOLS)} 个工具")


if __name__ == "__main__":
    main()
