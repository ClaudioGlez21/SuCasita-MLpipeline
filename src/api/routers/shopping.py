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