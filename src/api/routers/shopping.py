from fastapi import APIRouter
from src.api.schemas.schemas_shopping import CartItem

router = APIRouter(tags=["Shopping cart"])

router.get("/shopping-cart")

shopping = []

@router.get("/shopping-cart") 
def shopping_cart():
    '''Devuelve la lista de compras.'''
    return {"response": shopping}

@router.get("/shopping-cart/{item_id}")
def query_shopping_cart(item_id: int):
    '''
    Devuelve un elemento de la lista de compras.
    '''

    response = {"response": f"No se encontro el item {item_id} :()"} # <--- Rellenado

    ## Comprobamos que el id esté dentro de la lista (evitar negativos y fuera de rango)
    valid = item_id >= 0 and item_id < len(shopping) # <--- Rellenado

    ## Buscamos y devolvemos el item si es válido
    if valid: # <--- Rellenado
        response = {"item": shopping[item_id]} # <--- Rellenado

    return response

@router.post("/add-to-cart")
def add_to_cart(cart_item: CartItem):
    '''
    Añade un item a la lista de compras.

    Argumentos
    ----------
    cart_item: CartItem
        Elemento añadido a la lista.
    '''

    ## Añadimos el elemento item del
    ## objeto CartItem a la lista de compras
    shopping.append(cart_item.item)

    ## Generamos un objeto de respuesta
    response = {"response": f"¡Item {cart_item.item} añadido!"}

    return response


@router.delete("/delete-cart-item/{item_id}")
def delete_cart_item(item_id: int):
    '''
    Elimina un item de la lista de compras.
    
    Argumentos
    ----------
    item_id: int
        Posición del elemento a eliminar.
    '''

    ## Validar que la posición exista
    valid = 0 <= item_id < len(shopping)

    item = None
    ## Eliminar el item de la lista
    if valid:
        item = shopping[item_id]
        shopping.pop(item_id)

    ## Crear respuesta
    response = {"response": f"¡Item {item if valid else item_id} eliminado!"}

    return response


@router.put("/update-cart-item/{item_id}")
def update_cart_item(item_id: int, cart_item: CartItem):
    '''
    Actualiza un elemento de la lista de compras.

    Argumentos
    ----------
    item_id: int
        Posición del item que se reemplazará.
    cart_item: CartItem
        Valor con el que se reemplazará.
    '''

    ## Validar que la posición exista
    valid = 0 <= item_id < len(shopping)

    ## Realizar el reemplazo
    if valid:
        shopping[item_id] = cart_item.item

    ## Crear respuesta
    response = {"response": f"¡Item {item_id} actualizado con el valor {cart_item.item}!"}

    return response