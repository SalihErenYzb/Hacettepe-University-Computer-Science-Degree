import javafx.geometry.Pos;
import javafx.scene.Group;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.scene.text.TextAlignment;
/**

BackGroup class represents the background group of the game scene, which includes the background image, foreground image, and text instructions.
*/
public class BackGroup extends Group {
	private static ImageView imageView;

	/**
	 * Constructor for BackGroup class, creates 
	 * the crosshair changing group of the game scene.
	 */
	BackGroup(){
		ImagePane backPane = new ImagePane(DuckHunt.Backgrounds[DuckHunt.getI()]);
		ImagePane forePane = new ImagePane(DuckHunt.Foregrounds[DuckHunt.getI()]);
		ImageViewCreator();
		BorderPane borderPane = new BorderPane();
		Text text = new Text("USE ARROW KEYS TO NAVIGATE\nPRESS ENTER TO START\nPRESS ESC TO EXIT");
		text.setFont(Font.font("Arial", FontWeight.BOLD, 7*DuckHunt.SCALE));
		text.setTextAlignment(TextAlignment.CENTER);
		text.setFill(Color.rgb(255, 165, 0)); // Orange color
		
		BorderPane.setAlignment(text, Pos.BOTTOM_CENTER); // Center the text inside the BorderPane
		borderPane.setPrefSize(DuckHunt.SCALE*256, 240*DuckHunt.SCALE);
		//borderPane.setTop(text);
		borderPane.setTop(text);
		getChildren().addAll(backPane,forePane,borderPane,BackGroup.imageView);
	}
	public static void ImageViewCreator() {

		ImageView imageView = new ImageView(DuckHunt.Crosshairs[DuckHunt.getJ()]);


		imageView.setLayoutX(DuckHunt.SCALE*256/2-DuckHunt.Crosshairs[DuckHunt.getJ()].getWidth()/2);
        imageView.setLayoutY(DuckHunt.SCALE*240/2-DuckHunt.Crosshairs[DuckHunt.getJ()].getHeight()/2);

        BackGroup.imageView = imageView;
        
	}

}
